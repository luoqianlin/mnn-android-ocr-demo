package com.test.ocrdemo;

import org.opencv.core.MatOfPoint;
import org.opencv.core.RotatedRect;

class InnerBoxScore {
    //经过膨胀过后的区域
    public final RotatedRect rect;
    public final float score;
    //调试使用
    public final MatOfPoint contour;

    public InnerBoxScore(RotatedRect rect, float score,
                         MatOfPoint contour) {
        this.rect = rect;
        this.score = score;
        this.contour = contour;
    }

    @Override
    public String toString() {
        return "InnerBoxScore{" +
                "minAreaRect=" + rect +
                ", score=" + score +
                '}';
    }
}
