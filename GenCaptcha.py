from captcha.image import ImageCaptcha
import numpy as np
from PIL import Image
import pickle

TARGET_HEIGHT=48
TARGET_WIDTH=128
MAXLEN = 6


number = [str(i) for i in range(10)]
alphabet = [chr(i) for i in range(ord('a'),ord('z')+1)]
Alphabet = [chr(i) for i in range(ord('A'),ord('Z')+1)]
alphabet.remove('o')
Alphabet.remove('O')
charset = number+alphabet+Alphabet


def random_chars(charset, nb_chars):
    return [np.random.choice(charset) for i in range(nb_chars)]


def gen_captcha(charset,nb_chars=None,font=None):
    if not font is None:
        image = ImageCaptcha(fonts=[font])

    buffer_index=1000
    buffer_size=1000
    nc_set = np.zeros(buffer_size)


    while True:
        if buffer_index==buffer_size:
            nc_set = np.random.randint(3, MAXLEN+1, buffer_size) if nb_chars is None else np.array([nb_chars] * buffer_size)
            buffer_index=0
        captcha_text = ''.join(random_chars(charset,nc_set[buffer_index]))
        buffer_index+=1

        img_text = ' '*np.random.randint(0,MAXLEN+1-len(captcha_text))*2+captcha_text #用空格模拟偏移
        captcha = image.generate(img_text)
        captcha_image = Image.open(captcha).resize((TARGET_WIDTH,TARGET_HEIGHT),Image.ANTIALIAS)
        #image.write(captcha_text, captcha_text + '.jpg')  # 写到文件
        captcha_array = np.array(captcha_image)
        yield captcha_array,captcha_text


def convert_to_npz(num,captcha_generator,is_encoded,is_with_tags):
    vocab = charset[:]
    if is_encoded:
        vocab += [' ']
    if is_with_tags:
        id2token = {k+1:v for k,v in enumerate(vocab)}
        id2token[0] = '^'
        id2token[len(vocab)+1]='$'
    else:
        id2token = dict(enumerate(vocab))

    token2id = {v:k for k,v in id2token.items()}

    vocab_dict ={"id2token":id2token,"token2id":token2id}
    with open("data/captcha.vocab_dict","wb") as dict_file:
        pickle.dump(vocab_dict,dict_file)
    fn = "data/captcha.npz"

    print("Writing ",fn)
    img_buffer = np.zeros((num,TARGET_HEIGHT,TARGET_WIDTH,3),dtype=np.uint8)
    text_buffer = []
    for i in range(num):
        x,y = next(captcha_generator)
        img_buffer[i] = x
        if is_with_tags:
            y = ("^"+y+"$")
        if is_encoded:
            text_buffer.append([token2id[i] for i in y.ljust(MAXLEN+2*is_with_tags)])
        else:
            text_buffer.append(y)
    np.savez(fn,img=img_buffer,text=text_buffer)
    return vocab_dict,img_buffer,text_buffer


if __name__ == '__main__':
    captcha_generator = gen_captcha(charset,font='fonts/YaHeiConsolas.ttf') #生成器
    # x,y = next(captcha_generator)
    # plt.imshow(x)
    # plt.show()
    # print(y)
    vocab_dict,img,text = convert_to_npz(num=65536,captcha_generator=captcha_generator,
                    is_encoded=True,is_with_tags=True) #生成65536个样本
    #vocab_dict = convert_to_tfrecord(65536,captcha_generator,is_encoded=False,is_with_tags=True)

