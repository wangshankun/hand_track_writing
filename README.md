#摄像机跟踪手指的轨迹，识别轨迹中的数字
* 1.  hsv域的肤色抠图
* 2. 腐蚀之后踢掉零星碎点
* 3. 转为binary图
* 4. 使用np.nonzero，找出最上方的坐标点，也就是伸出手指的指尖的坐标
* 5. 使用list记录坐标，根据记录信息去可以判断轨迹开始，结束···
     hit_contours.append((tx, ty))
     hit_contours.count
* 6. opencv 的全屏显示设置cv2.setWindowProperty
* 7. opencv 不同按键的响声实现
* 8. PIL Image 与 opencv 格式互相转换
* 9. PIL Image 实现的图像拼接，复制
* 10.  opencv的轮廓寻找cv2.findContours，根据轮廓大小，剔除杂点
* 11.  opencv的两图像直接的 与，或，且，非，异或操作，cv2.bitwise_and，方便用来扣图(二值图合并)
* 12. 如果二值图，比灰度图的特征更好，那么二值图的准确率会很高
