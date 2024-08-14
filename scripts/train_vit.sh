# python ../src/train_timm_dns.py -c ./config/vit.yml --dns-ratio 0.5 --use-dns 0 --use-fr 0 --use-kd 0 --use-fd 0 --experiment "vit-tiny-without-dns-AdamW-v4"
# python ../src/train_timm_dns.py -c ./config/vit.yml --dns-ratio 0.5 --use-dns 0 --use-fr 0 --use-kd 0 --use-fd 0 --experiment "vit-tiny-without-dns-AdamW-v2"
# python ../src/train_timm_dns.py -c ./config/vit.yml --dns-ratio 0.1 --use-dns 1 --use-fr 1 --use-kd 0 --use-fd 0 --experiment "vit-tiny-with-dns0.1-AdamW-v1"
# python ../src/train_timm_dns.py -c ./config/vit.yml --dns-ratio 0.1 --use-dns 1 --use-fr 1 --use-kd 0 --use-fd 0 --experiment "vit-tiny-with-dns0.1-AdamW-v4"
# python ../src/train_timm_dns.py -c ./config/vit.yml --dns-ratio 0.3 --use-dns 1 --use-fr 1 --use-kd 0 --use-fd 0 --experiment "vit-tiny-with-dns0.3-AdamW-v4"
# python ../src/train_timm_dns.py -c ./config/vit.yml --dns-ratio 0.5 --use-dns 1 --use-fr 1 --use-kd 0 --use-fd 0 --experiment "vit-tiny-with-dns0.5-AdamW-v4"
# python ../src/train_timm_dns.py -c ./config/vit.yml --dns-ratio 0.7 --use-dns 1 --use-fr 1 --use-kd 0 --use-fd 0 --experiment "vit-tiny-with-dns0.7-AdamW-v4"
# python ../src/train_timm_dns.py -c ./config/vit.yml --dns-ratio 0.9 --use-dns 1 --use-fr 1 --use-kd 0 --use-fd 0 --experiment "vit-tiny-with-dns0.9-AdamW-v4"

# run w/o dfs 0.1 0.3 results
python ../src/train_timm_dns.py -c ./config/vit.yml --dns-ratio 0.5 --use-dns 0 --use-fr 0 --use-kd 0 --use-fd 0 --experiment "vit-tiny-without-dns0.5-AdamW-v4"
python ../src/train_timm_dns.py -c ./config/vit.yml --dns-ratio 0.3 --use-dns 1 --use-fr 1 --use-kd 0 --use-fd 0 --experiment "vit-tiny-with-dns0.3-AdamW-v4"
python ../src/train_timm_dns.py -c ./config/vit.yml --dns-ratio 0.1 --use-dns 1 --use-fr 1 --use-kd 0 --use-fd 0 --experiment "vit-tiny-with-dns0.1-AdamW-v4"
