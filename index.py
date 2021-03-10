# -*- coding: utf-8 -*-

import cv2
import dlib
import base64
import json
import uuid
import bottle

app = bottle.default_app()

predictorPath = "shape_predictor_5_face_landmarks.dat"
predictor = dlib.shape_predictor(predictorPath)
detector = dlib.get_frontal_face_detector()

return_msg = lambda error, msg: {
    "uuid": str(uuid.uuid1()),
    "error": error,
    "message": msg
}


def addHat(img, hat_img):
    # 分离rgba通道，合成rgb三通道帽子图，a通道后面做mask用
    r, g, b, a = cv2.split(hat_img)
    rgbHat = cv2.merge((r, g, b))

    # dlib人脸关键点检测器,正脸检测
    dets = detector(img, 1)

    # 如果检测到人脸
    if len(dets) > 0:
        for d in dets:
            x, y, w, h = d.left(), d.top(), d.right() - d.left(), d.bottom() - d.top()

            # 关键点检测，5个关键点")
            shape = predictor(img, d)

            # 选取左右眼眼角的点")
            point1 = shape.part(0)
            point2 = shape.part(2)

            # 求两点中心
            eyes_center = ((point1.x + point2.x) // 2, (point1.y + point2.y) // 2)

            # 根据人脸大小调整帽子大小
            factor = 1.5
            resizedHatH = int(round(rgbHat.shape[0] * w / rgbHat.shape[1] * factor))
            resizedHatW = int(round(rgbHat.shape[1] * w / rgbHat.shape[1] * factor))

            if resizedHatH > y:
                resizedHatH = y - 1

            # 根据人脸大小调整帽子大小
            resizedHat = cv2.resize(rgbHat, (resizedHatW, resizedHatH))

            # 用alpha通道作为mask
            mask = cv2.resize(a, (resizedHatW, resizedHatH))
            maskInv = cv2.bitwise_not(mask)

            # 帽子相对与人脸框上线的偏移量
            dh = 0
            bgRoi = img[y + dh - resizedHatH:y + dh,
                    (eyes_center[0] - resizedHatW // 3):(eyes_center[0] + resizedHatW // 3 * 2)]

            # 原图ROI中提取放帽子的区域
            bgRoi = bgRoi.astype(float)
            maskInv = cv2.merge((maskInv, maskInv, maskInv))
            alpha = maskInv.astype(float) / 255

            # 相乘之前保证两者大小一致（可能会由于四舍五入原因不一致）
            alpha = cv2.resize(alpha, (bgRoi.shape[1], bgRoi.shape[0]))
            bg = cv2.multiply(alpha, bgRoi)
            bg = bg.astype('uint8')

            # 提取帽子区域
            hat = cv2.bitwise_and(resizedHat, cv2.bitwise_not(maskInv))

            # 相加之前保证两者大小一致（可能会由于四舍五入原因不一致）")
            hat = cv2.resize(hat, (bgRoi.shape[1], bgRoi.shape[0]))
            # 两个ROI区域相加")
            addHat = cv2.add(bg, hat)

            # 把添加好帽子的区域放回原图
            img[y + dh - resizedHatH:y + dh,
            (eyes_center[0] - resizedHatW // 3):(eyes_center[0] + resizedHatW // 3 * 2)] = addHat

            return img


@bottle.route('/add/hat', method='POST')
def addHatIndex():
    try:
        try:
            # 将接收到的base64图像转为pic
            postData = json.loads(bottle.request.body.read().decode("utf-8"))
            imgData = base64.b64decode(postData.get("image", None))
            with open('/tmp/picture.png', 'wb') as f:
                f.write(imgData)
        except Exception as e:
            print(e)
            return return_msg(True, "未能成功获取到头像，请检查pic参数是否为base64编码。")

        try:
            # 读取帽子素材以及用户头像
            hatImg = cv2.imread("hat.png", -1)
            userImg = cv2.imread("/tmp/picture.png")

            output = addHat(userImg, hatImg)
            cv2.imwrite("/tmp/output.jpg", output)
        except Exception as e:
            return return_msg(True, "图像添加圣诞帽失败，请检查图片中是否有圣诞帽或者图片是否可读。")

        # 读取头像进行返回给用户，以Base64返回
        with open("/tmp/output.jpg", "rb") as f:
            base64Data = str(base64.b64encode(f.read()), encoding='utf-8')

        return return_msg(False, {"picture": base64Data})
    except Exception as e:
        return return_msg(True, str(e))
