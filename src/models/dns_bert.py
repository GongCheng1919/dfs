import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
from transformers import BertForSequenceClassification
from transformers.models.bert.modeling_bert import BertEncoder,BertPooler
from transformers.modeling_outputs import SequenceClassifierOutput,BaseModelOutputWithPastAndCrossAttentions
import time

class FeatureReroute(nn.Module):
    '''
    Split the feature into two parts with the ratio along channel dimension,
    '''
    def __init__(self, num_channel, ratio=0.5):
        super().__init__()
        self.ratio = max(min(ratio,1),0)
        self.num_channel = num_channel
        self.feature_num1 = round(num_channel*ratio)
        self.feature_num2 = num_channel - self.feature_num1

    def forward(self,x): # 默认为nchw维度，对c维度进行切分
        # 特征分割
        tmp1, tmp2 = torch.split(x,[self.feature_num1,self.feature_num2],dim=-1)
        # 特征重组
        x_plus = torch.cat([tmp1,tmp2.detach()],-1)
        x_sub = torch.cat([tmp1.detach(),tmp2],-1)
        # print(f"Using dns ratio is {self.ratio}. The shared featrue number is {self.feature_num1} and unshared one is {self.feature_num2}")
        return x_plus,x_sub

def feature_reroute_func(x, ratio=0.5):
    return FeatureReroute(x.size(-1), ratio)(x)

# If we want to change the split path, we must rewrite the BertEncoder
class DNSBertEncoder(BertEncoder):
    def __init__(self, config, dns_ratio=0.5):
        super().__init__(config)
        self.dns_ratio=dns_ratio

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        next_decoder_cache = () if use_cache else None
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                if i==0:
                    all_hidden_states = all_hidden_states + (hidden_states,)
                else:
                    x_plus,x_sub = feature_reroute_func(hidden_states,self.dns_ratio)
                    hidden_states = x_plus
                    all_hidden_states = all_hidden_states + (x_sub,)


            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            if getattr(self.config, "gradient_checkpointing", False) and self.training:

                if use_cache:
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, past_key_value, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                )

            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )

class MTLBertForSequenceClassification(BertForSequenceClassification):
    def __init__(self, config,
           use_dns = False,
           use_fr = False,
           dns_ratio = 0.5):
        super().__init__(config)
        # 最后一层是不必要的，因为会默认加，所以只加前11层
        self.middle_pooler_layers = nn.ModuleList([BertPooler(config) for _ in range(config.num_hidden_layers-1)])
        self.middle_dropuout_layers = nn.ModuleList([nn.Dropout(config.hidden_dropout_prob) for _ in range(config.num_hidden_layers-1)])
        self.middle_classifier_layers = nn.ModuleList([nn.Linear(config.hidden_size, config.num_labels) for _ in range(config.num_hidden_layers-1)])
        if use_dns:
          self.bert.encoder = DNSBertEncoder(self.bert.encoder.config,dns_ratio=dns_ratio)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=True, # 默认返回每层hidden_states，用于中间层分类
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # middle tasks: there are total 13 hidden states and the first is embbeding states and last is final outputs, we only use the middle ones
        middle_hidden_states = outputs.hidden_states
        middle_logits = []
        middle_losses = []
        for i in range(len(self.middle_pooler_layers)):
          pooled_output = self.middle_pooler_layers[i](middle_hidden_states[i+1])
          pooled_output = self.middle_dropuout_layers[i](pooled_output)
          logits = self.middle_classifier_layers[i](pooled_output)
          middle_logits.append(logits)
          loss = None
          if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
          middle_losses.append(loss)

        # final tasks
        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = tuple(middle_logits)+(logits,) + outputs[2:]
            return (tuple(middle_losses)+(loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=tuple(middle_losses)+(loss,),
            logits=tuple(middle_logits)+(logits,),
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
