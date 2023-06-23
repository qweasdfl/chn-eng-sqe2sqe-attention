import torch
from chinese_datapre import prepareData
from seq2seq_model import EncoderRNN, AttnDecoderRNN
from evaluate import evaluateRandomly, evaluate, showAttention
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义一个变量来保存加载后的模型和参数
model = None
# 判断当前是否支持 GPU 加速
if torch.cuda.is_available():
    # 如果支持 GPU 加速，则加载在 GPU 上保存的模型和参数
    model = torch.load('models/encoder.pkl')
else:
    # 如果不支持 GPU 加速，则加载在 CPU 上保存的模型和参数
    model = torch.load('models/decoder.pkl', map_location=torch.device('cpu'))

# 进行后续操作（比如预测、训练等）

'''
model = torch.load('models/encoder.pkl', map_location=torch.device('cpu'))
# 使用 cpu()
device = torch.device('cpu')
model.to(device)  # 将该模型装载到CPU上
'''

def evaluateAndShowAttention(input_sentence):
    output_words, attentions = evaluate(input_lang, output_lang, encoder1, attn_decoder1, input_sentence)
    print('input =', input_sentence)
    print('output =', ' '.join(output_words))
    showAttention(input_sentence, output_words, attentions)


if __name__ == '__main__':
    input_lang, output_lang, pairs = prepareData('eng', 'chin', True)
    hidden_size = 256  # 隐藏层维度设置为256
    encoder1 = EncoderRNN(input_lang.n_words, hidden_size).to(device)
    attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.1).to(device)
    # 恢复网络
    encoder1.load_state_dict(torch.load('models/encoder.pkl'))
    attn_decoder1.load_state_dict(torch.load('models/decoder.pkl'))

    # 随机从数据集选几个句子翻译下
    evaluateRandomly(input_lang, output_lang, pairs, encoder1, attn_decoder1)

    a=1
    while  True:
         aa = input("请输入要翻译的句子：")

        # 输入句子测试下，句子的单词必须是数据集里有的，否则报错
         evaluateAndShowAttention(aa)
         a+=1
