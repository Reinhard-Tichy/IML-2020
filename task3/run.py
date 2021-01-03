# coding: UTF-8
import time
from importlib import import_module
import argparse
from config.config import BertConfig
from train import predict, train, get_time_dif

parser = argparse.ArgumentParser(description='iMDB sentiment analysis')
parser.add_argument("-p","--predict", action="store_true", help = 'predict the test')
parser.add_argument("-m","--model", type=str, default="Bert")
args = parser.parse_args()

if __name__ == '__main__':
    x = import_module('models.Bert')
    config = BertConfig()
    model = x.Model(config, config.model_name).to(config.device)
    #np.random.seed(1)
    #torch.manual_seed(1)
    #torch.cuda.manual_seed_all(1)
    #torch.backends.cudnn.deterministic = True 
    start_time = time.time()
    print("Loading data...")
    if args.predict:
        test_iter = config.get_predict_iter()
        time_dif = get_time_dif(start_time)
        print("Time usage:", time_dif)
        predict_all = predict(config, model, test_iter)
        print("predict finished, store the results")
        config.gen_submit(predict_all)
    else :
        train_iter, dev_iter = config.get_train_eval_test_iter()
        time_dif = get_time_dif(start_time)
        print("Time usage:", time_dif)
        train(config, model, train_iter, dev_iter)
        
