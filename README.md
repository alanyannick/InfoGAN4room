# InfoGAN4room
This is the git for bedroom experiments 

主代碼在develop branch: [develop](https://github.com/alanyannick/InfoGAN4room/tree/develop)
拉取codes方式: 在terminal中key in:

```git clone -b develop https://github.com/alanyannick/InfoGAN4room.git```

## 訓練模型階段:
運行 /run_InfoGAN.py 並在```run_InfoGAN```中設定參數。

`` 
run_InfoGAN(n_conti=2, n_discrete=1, D_featmap_dim=64, G_featmap_dim=128,batch_size=10,
                # revise the following to control training process
                n_epoch=10, update_max=5000,use_gpu=True,
                # 10 x 500 = 5000 iretation
save_experiments_folder='./experiments/',save_model_folder='./models/', model_choice='train')
``    



### 主要參數為 
* n_conti=2, n_discrete=1 對應於測試階段conti_codes 和 discr_codes的array緯度。
* save_experiments_folder, save_model_folder 保存模型的路徑 和 保存實驗結果的路徑

## 測試模型階段:
運行 /test_InfoGAN.py 主要動手修改這邊的 conti_codes 和 discr_codes， 觀察這些值的變化會對輸出產生什麼影響。

### 主要參數為 
* 測試階段conti_codes 和 discr_codes的array緯度。
* save_model_folder='./models/', test_model_name='7_checkpoint.pth', 即需要測試 訓練階段保存模型的路徑
* 按esc可依次看到結果show。
