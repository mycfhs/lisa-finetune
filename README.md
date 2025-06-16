## 在自己数据集上微调 LISA

有任何问题可以提issue！

### env (在原来lisa的环境基础上)
torch2.1.0+cuda121
pip install flash-attn==2.1.0 --no-build-isolation
pip install bleach

pip install tensorboard deepspeed scikit-image

### 依赖
apt install libaio-dev

### 遇到的问题
我记得主要是微调后的模型推理有问题 效果极差；原因在于训练的时候model.model.visual_model和一个model.model.text_hidden_fcs没有load进去，所以可以参考infer_crop.py line153-154 先跑一遍infer_crop.py把这俩预训练权重保存一下，然后训练的时候加载一遍。 训练加载

### 对源码进行的修改:

1. https://github.com/dvlab-research/LISA/issues/85    修改了 model/LISA.py  line135
2. 修改了 merge_lora_weights_and_save_hf_model.py， 改为支持微调的.bin权重加载模型    line146
3. 修改了 model/LISA.py  line100 读取fc权重 是专门为了训练加载的
4. 修改了 model/segment_anything/build_sam.py line108
5. 修改了train_ds.py  bs->1 line68 , line388强制每个epoch都保存权重

### 训练指令
参考train.sh。 训练数据只有reason_seg。 --vision_pretrained参数就是前面手动保存的其中一个模型参数。 推理代码infer.py是参考chat.py改的 主要是改为我们数据的批量跑图 没有大改动。
  
