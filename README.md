# train-qwen2-category  
使用Qwen2-1.5b-Instruct模型在zh_cls_fudan_news数据集上进行指令微调任务，同时使用SwanLab进行监控和可视化

## 1.环境安装  
本案例基于Python>=3.10.0，请在您的计算机上安装好Python，并且有一张英伟达显卡（显存要求并不高，大概10GB左右就可以跑）。  
pip install swanlab modelscope transformers datasets peft pandas accelerate   
本案例测试于modelscope1.14.0、transformers4.41.2、datasets2.18.0、peft0.11.1、accelerate0.30.1、swanlab0.3.9   

重要依赖安装torch-GPU系列：   
torch-2.2.1+cu121-cp310-cp310-win_amd64.whl  
torchaudio-2.2.1+cu121-cp310-cp310-win_amd64.whl  
torchvision-0.17.1+cu121-cp310-cp310-win_amd64.whl  

## 2. 数据集准备  
[开放域的分类问题](https://modelscope.cn/datasets/swift/zh_cls_fudan-news/dataPeview)  
```
    from modelscope import MsDataset
    dataset = MsDataset.load('swift/zh_cls_fudan-news', split='train')
    test_dataset = MsDataset.load('swift/zh_cls_fudan-news', subset_name='test', split='test')
    print(dataset)
    print(test_dataset)
    """
    Dataset({
        features: ['text', 'category', 'output'],
        num_rows: 4000
    })
    Dataset({
        features: ['text', 'category', 'output'],
        num_rows: 959
    })
    """
```

## 3. 训练过程图表    
 [训练数据指标看板](https://swanlab.cn/@ai-next-furture/train-qwen2-category/runs/kdcee8x3wns9060pqzssy/chart)   


## 4. 相关链接  
实验日志过程：Qwen2-1.5B-Fintune - SwanLab  
模型：Modelscope  
数据集：[zh_cls_fudan_news](https://www.modelscope.cn/datasets/swift/zh_cls_fudan-news)  
SwanLab：[swanlab实验看板](https://swanlab.cn)  
