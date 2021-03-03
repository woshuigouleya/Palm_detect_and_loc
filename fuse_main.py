import Palm_correction as four
import Palm_key_point_positioning as three
import make_data as second
import Palm_area_detection as first
import argparse
import cv2
import os
import numpy as np
import sys


def my_resize(img, save_path, origin_save_path):
    '''
    :param img:
    :param save_path:
    :param origin_save_path:
    :return:
    '''
    if img.shape[0] < img.shape[1]:  # h<w
        # cv2.circle(img, (int(img.shape[1] * 0.5), int(img.shape[0] * 0.5)), 4, (0, 255, 255), 100)
        rotate = cv2.getRotationMatrix2D((img.shape[0] * 0.5, img.shape[0] * 0.5), -90, 1)
        origin_img = cv2.warpAffine(img, rotate, (img.shape[0], img.shape[1]))
        # my_cv_imwrite(origin_save_path, origin_img)
        # print('true')
    else:
        origin_img = img
    img = cv2.resize(origin_img, (192, 192))
    my_cv_imwrite(save_path + '//' + 'resize_192.jpg', img)
    return origin_img, img


def my_cv_imread(filepath):
    # 使用imdecode函数进行读取
    img = cv2.imdecode(np.fromfile(filepath, dtype=np.uint8), -1)
    return img


def my_cv_imwrite(filepath, img):
    # 使用imencode函数进行读取
    cv2.imencode('.jpg', img)[1].tofile(filepath)


def crop_one(img, area, save_path, pad):
    x1, y1, x2, y2 = area
    pad_x, pad_y = pad
    x1 = int(img.shape[1] * x1)
    y1 = int(img.shape[0] * y1)
    x2 = int(img.shape[1] * x2)
    y2 = int(img.shape[0] * y2)

    y1 = int(y1 - pad_y) if y1 - pad_y > 0 else 0
    x1 = int(x1 - pad_x) if x1 - pad_x > 0 else 0
    y2 = int(y2 + pad_y) if y2 + pad_y < img.shape[0] else img.shape[0]
    x2 = int(x2 + pad_x) if x2 + pad_x < img.shape[1] else img.shape[1]
    img = img[y1:y2, x1:x2, :]

    img = cv2.resize(img, (192, 192))
    my_cv_imwrite(save_path + '//' + 'crop_one.jpg', img)
    return img, [x1, y1, x2, y2]


def advance_judge(img):
    '''
    3024*4032=(75,100)
    897*1920=(25,50)
    判断手掌的分辨率是否合格，并且确定padding大小
    :param img:
    :return:
    '''
    if img.shape[0] < 1500 and img.shape[1] < 1500:
        print("图片分辨率过低，请重新拍摄")
        sys.exit()
    else:
        pad_x = img.shape[0] / 40
        pad_y = img.shape[1] / 40
        # print(pad_x, pad_y)
    return (pad_x, pad_y)


def describution(pred):
    # 便秘，胃十二指肠炎，胃神经功能症，附件炎，卵巢囊肿，乳腺增生，子宫肌瘤，胆结石，胆囊炎，慢性肝炎，脂肪肝，男性功能性障碍，前列腺炎，头晕，头痛，失眠，近视，咽炎，中耳炎耳鸣
    title = ['便秘', '胃十二指肠炎', '胃神经功能症', '附件炎', '卵巢囊肿', '乳腺增生', '子宫肌瘤', '胆结石', '胆囊炎', '慢性肝炎', '脂肪肝', '男性功能性障碍', '前列腺炎',
             '头晕', '头痛', '失眠', '近视', '咽炎', '中耳炎耳鸣']
    des_title = ['【分析结果】', '您有极大可能患有', '除此之外，掌纹信息中还反映出您有', '可能患有',
                 '的趋势']  # 有极大可能患有XXX，除此之外，掌纹信息中还反映出您有可能患有XXX和XXX的趋势
    des_big = []
    des_medium = []
    des_small = []
    for i in range(len(pred)):
        if pred[i] > 0.6:
            des_big.append(title[i])
        elif pred[i] > 0.3 and pred[i] <= 0.6:
            des_medium.append(title[i])
        elif pred[i] > 0.1 and pred[i] <= 0.3:
            des_small.append(title[i])
    des = '【分析结果】'
    if len(des_big) == 0 and len(des_medium) == 0 and len(des_small) == 0:
        des += '您的预检结果体现良好，但健康不容忽视，还是建议您定期去医院体检，将一切趋势扼杀在摇篮里。祝您身体健康。\n'
    else:
        if len(des_big) != 0:
            des += '您有极大可能患有'
            for i in range(len(des_big)):
                des += des_big[i]
                if i < len(des_big) - 2:
                    des += '、'
                elif i == len(des_big) - 2:
                    des += '和'
                else:
                    None
            des += '。\n'
        if len(des_medium) != 0 or len(des_small) != 0:
            if len(des_big) != 0:
                des += '除此之外，掌纹信息中还反映出您'
            else:
                des += '您在我们预测的疾病中，并未存在极大概率患有的疾病。\n但健康不容忽视，掌纹中还反映出一些其他信息，您'
            if len(des_medium) != 0:
                des += '有一定可能患有'
                for i in range(len(des_medium)):
                    des += des_medium[i]
                    if i < len(des_medium) - 2:
                        des += '、'
                    elif i == len(des_medium) - 2:
                        des += '和'
                    else:
                        None
                if len(des_small) != 0:
                    des += ',并存在极小可能患有'
                    for i in range(len(des_small)):
                        des += des_small[i]
                        if i < len(des_small) - 2:
                            des += '、'
                        elif i == len(des_small) - 2:
                            des += '和'
                        else:
                            None
                des += '。\n'
            else:
                if len(des_small) != 0:
                    des += '存在极小可能患有'
                    for i in range(len(des_small)):
                        des += des_small[i]
                        if i < len(des_small) - 2:
                            des += '、'
                        elif i == len(des_small) - 2:
                            des += '和'
                        else:
                            None
                    des += '。\n'
    des += '该结果由我们设计的AI算法，通过超过3万掌纹数据自动学习得出。以上预检结果并非绝对，但仍存在较高可信度。'
    if len(des_big) != 0 or len(des_medium) != 0 or len(des_small) != 0:
        des += '健康为先，您可以到权威医院进一步进行检查。'
    return des


if __name__ == '__main__':
    '''
    zero把图片resize到192
    first用于手掌区域检测，输入192*192的图，输出标记框的图和label.txt
    second用于把检测结果的xywh转换为xyxy，输入xywh的label.txt，输出xyxy的label.txt
    three用于关键点定位，输入原图中根据label.txt中坐标padding100个像素之后再resize到192的图，输出12个关键点
    four用于根据关键点定位后的结果进行矫正，输入结果图、原图，输出512*512的矫正图
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='ls2.jpg', help='name')
    parser.add_argument('--origin_source', type=str, default='data', help='source')
    parser.add_argument('--source', type=str, default='', help='source')
    parser.add_argument('--weight_one', nargs='+', type=str, default='best.pt', help='model.pt path(s)')
    parser.add_argument('--img-size', type=int, default=192, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_false', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_false', help='existing project/name ok, do not increment')
    parser.add_argument('--weight_two', nargs='+', type=str, default='min_palm_model_epoch.pt', help='model.pt path(s)')
    opt = parser.parse_args()

    # origin_img = my_cv_imread(os.path.abspath(opt.origin_source) + '//' + opt.name)
    # padding = advance_judge(origin_img)
    # opt.source = os.path.abspath(opt.project) + '//' + opt.name
    # if not os.path.exists(opt.source):
    #     os.makedirs(opt.source)
    #
    # origin_img, img = my_resize(origin_img, opt.source, opt.origin_source + '//' + opt.name)
    # txt_path = first.detect(opt)
    # area = second.xywh2xyxy_(txt_path)
    # img, area = crop_one(origin_img, area, opt.source, padding)
    #
    # test = three.PalmTest(image_dir=opt.source, o_net_path=opt.weight_two)
    # landmarks = [[float(test.landmarks[2 * i]), float(test.landmarks[2 * i + 1])] for i in range(12)]
    # four.correction_(origin_img, opt.source, landmarks, area, padding)


    pred = [0.02942, 0.07218, 0.09929, 0.0004275, 0.0057693, 0.00094291, 0.0067087, 0.036022, 0.02054, 0.00058,
            0.02701, 0.02116, 0.02121, 0.051721, 0.0062477, 0.051561, 0.00097198, 0.0175, 0.013663]
    des = describution(pred)
    print(des)
