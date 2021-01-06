package com.test.ocrdemo;

import org.opencv.core.Point;
import org.opencv.core.Rect;

public class BoxScore {
    public final Rect box;
    public final float score;
    public final double angle;
    public final Point center;


    public BoxScore(Rect box, float score, double angle, Point center) {
        this.box = box;
        this.score = score;
        this.angle = angle;
        this.center = center;
    }

    @Override
    public String toString() {
        return "BoxScore{" +
                "box=" + box +
                ", score=" + score +
                ", angle=" + angle +
                ", center=" + center +
                '}';
    }
}
