python test.py --name face-link-neck-v2_finetun --label_nc 0  --dataroot /sdf_data/liyao/Database/YZMHeadMattingselectImg/trainSet/ --no_instance --loadSize 512 --gpu_ids 1 --tf_log --resize_or_crop none --phase val --which_epoch latest --export_onnx ./pix2pix.onnx