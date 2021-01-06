package com.test.ocrdemo;

import android.graphics.Bitmap;
import android.util.Log;

import com.clipper.ClipperLib;

import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.RotatedRect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class MNNOCRDetPostProcess {
    private static final String TAG = "MNNOCRDetPostProcess";
    //概率最小阈值
    private float thresh = 0.1f;
    //文本区域最小平均概率
    private float box_thresh = 0.5f;
    //最多能够检测的文本区域个数,未用
    private int max_candidates = 1000;
    //检测到多边形扩大因子
    private float unclip_ratio = 2.0f;
    //文本区域最小边长阈值
    private int min_size = 4;

    private boolean showLog = false;

    private Bitmap detectedResult;

    public void setShowLog(boolean showLog) {
        this.showLog = showLog;
    }

    public boolean isShowLog() {
        return showLog;
    }

    /**
     * @param dimensions    预测结果的维度信息
     * @param predictResult 预测结果
     * @param ratioHeight   高度缩放比例
     * @param ratioWidth    宽度缩放比例
     * @param rawHeight     原始图像高度
     * @param rawWidth      原始图像宽度
     */
    public List<BoxScore> process(int[] dimensions, float[] predictResult,
                                  float ratioHeight, float ratioWidth,
                                  int rawHeight, int rawWidth) {

        float[] predData = new float[dimensions[2] * dimensions[3]];
        System.arraycopy(predictResult, 0, predData, 0, predData.length);
        MatOfFloat mat = new MatOfFloat();
        mat.fromArray(predData);
        Mat pred = mat.reshape(0, dimensions[2]);
        Mat segmentation = new Mat();
        Core.compare(pred, new Scalar(this.thresh), segmentation, Core.CMP_GT);
        mat.release();

        List<InnerBoxScore> innerBoxScores = findBoxs(pred, segmentation);
        pred.release();

        List<BoxScore> boxScores = revertBox(ratioHeight, ratioWidth,
                rawHeight, rawWidth, innerBoxScores);

        drawDetectedResult(segmentation, innerBoxScores);

        release(innerBoxScores);

        segmentation.release();

        return boxScores;
    }

    /**
     * 将坐标还原为原图上的坐标
     */
    private List<BoxScore> revertBox(float ratioHeight, float ratioWidth,
                                     int rawHeight, int rawWidth,
                                     List<InnerBoxScore> innerBoxScores) {

        List<BoxScore> boxScores = new ArrayList<>();
        for (InnerBoxScore innerBoxScore : innerBoxScores) {
            if (innerBoxScore.rect.size.width / ratioWidth <= 10
                    || innerBoxScore.rect.size.height / ratioHeight <= 10) {
                continue;
            }
            Rect rect = innerBoxScore.rect.boundingRect();
            Point center = innerBoxScore.rect.center;
            int x = clip((int) (rect.x / ratioWidth), 0, rawWidth);
            int y = clip((int) (rect.y / ratioHeight), 0, rawHeight);
            int w = clip((int) (rect.width / ratioWidth), 0, rawWidth);
            int h = clip((int) (rect.height / ratioHeight), 0, rawHeight);
            int centerX = clip((int) (center.x / ratioWidth - x), 0, rawWidth);
            int centerY = clip((int) (center.y / ratioHeight - y), 0, rawHeight);
            Point newCenter = new Point(centerX, centerY);
            BoxScore box = new BoxScore(new Rect(x, y, w, h),
                    innerBoxScore.score, innerBoxScore.rect.angle, newCenter);
            boxScores.add(box);
        }
        return boxScores;
    }

    private void release(List<InnerBoxScore> innerBoxScores) {
        for (InnerBoxScore innerBoxScore : innerBoxScores) {
            innerBoxScore.contour.release();
        }
    }

    /**
     * 将检测到的文字区域绘制到detectedResult，主要用于调试
     */
    private void drawDetectedResult(Mat segmentation, List<InnerBoxScore> innerBoxScores) {
        final int height = segmentation.rows();
        final int width = segmentation.cols();
        Mat canvas = new Mat(segmentation.rows(), segmentation.cols(), CvType.CV_8UC3, new Scalar(0));
        List<MatOfPoint> contours = getContours(innerBoxScores);
        Imgproc.drawContours(canvas, contours, -1, new Scalar(255), 4);
        if (detectedResult == null
                || detectedResult.getWidth() != width || detectedResult.getHeight() != height) {
            if (detectedResult != null
                    && !detectedResult.isRecycled()) {
                detectedResult.recycle();
            }
            detectedResult = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);
        }
        Utils.matToBitmap(canvas, detectedResult);
        canvas.release();
    }

    public static Mat submat(Mat mat, BoxScore bbox) {
        Mat img = mat.submat(bbox.box);
        if (true) {
            return img;
        }
//        Point point=new Point(bbox.center.x-10,bbox.center.y-10);
//        Imgproc.rectangle(img,bbox.center,point,new Scalar(0,0,255),2);
        double angle = bbox.angle;
        if (angle < -45) {
            angle = -(90 + angle);
        } else {
//            angle=-angle;
        }
        Log.d(TAG, "angle:" + angle);
        if (Double.compare(bbox.angle, 0.0) == 0) {
            return img;
        }
        Size srcSize = img.size();
        Mat matrix2D = Imgproc.getRotationMatrix2D(bbox.center, angle, 1.0);
        Mat rotated = new Mat();
        Imgproc.warpAffine(img, rotated, matrix2D, srcSize, Imgproc.INTER_CUBIC);
        matrix2D.release();
        img.release();
        return rotated;
    }

    private List<MatOfPoint> getContours(List<InnerBoxScore> innerBoxScores) {
        List<MatOfPoint> contours = new ArrayList<>();
        for (InnerBoxScore innerBoxScore : innerBoxScores) {
            contours.add(innerBoxScore.contour);
        }
        return contours;
    }


    public List<InnerBoxScore> findBoxs(Mat pred, Mat bitmap) {
        List<MatOfPoint> contours = new ArrayList<>();
        Mat hierarchy = new Mat();
        Imgproc.findContours(bitmap, contours, hierarchy, Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE);
        int num_contours = Math.min(contours.size(), this.max_candidates);
        List<InnerBoxScore> innerBoxScores = new ArrayList<>();
        if (showLog) {
            Log.d(TAG, "find contours size:" + contours.size());
        }
        for (int i = 0; i < contours.size(); i++) {
            MatOfPoint contour = contours.get(i);
            RotatedRect miniBox = getMinAreaRect(contour);
            double sside = getMinEdgeLength(miniBox);
            if (sside < this.min_size) {
                contour.release();
                continue;
            }
            double score = boxScoreFast(pred, miniBox);
            if (score < this.box_thresh) {
                contour.release();
                continue;
            }
            RotatedRect dilateRect = dilatePolygon(miniBox);
            if (getMinEdgeLength(dilateRect) < this.min_size + 2) {
                contour.release();
                continue;
            }
            innerBoxScores.add(new InnerBoxScore(dilateRect, (float) score, contour));
        }
        if (showLog) {
            Log.d(TAG, "boxes:" + innerBoxScores.size() + ":" + innerBoxScores);
        }
        hierarchy.release();
        return innerBoxScores;
    }

    //扩大检测到的文字区域
    private RotatedRect dilatePolygon(RotatedRect rect) {
        float[][] points = getMinAreaRectPoints(rect);
        double distance = getCountourDistance(points);
        float[][][] offsetPoints = ClipperLib.clipperOffset(points, distance);
        if (showLog) {
            Log.d(TAG, "offsetPoints: (" + offsetPoints.length + "," + offsetPoints[0].length + "," + offsetPoints[0][0].length + ")");
        }
        MatOfPoint mat = new MatOfPoint();
        List<Point> points1 = new ArrayList<>();
        for (int i = 0; i < offsetPoints.length; i++) {
            for (int k = 0; k < offsetPoints[i].length; k++) {
                float x = offsetPoints[i][k][0];
                float y = offsetPoints[i][k][1];
                points1.add(new Point(x, y));
            }
        }
        mat.fromList(points1);
        RotatedRect minAreaRect = getMinAreaRect(mat);
        mat.release();
        return minAreaRect;
    }

    //计算扩大值,扩大值=面积/周长×扩大因子
    private double getCountourDistance(float[][] box) {
        int pts_num = 4;
        double area = 0;
        double dist = 0;
        for (int i = 0; i < pts_num; i++) {
            area += box[i][0] * box[(i + 1) % pts_num][1] - box[i][1] * box[(i + 1) % pts_num][0];
            dist += Math.sqrt((box[i][0] - box[(i + 1) % pts_num][0]) *
                    (box[i][0] - box[(i + 1) % pts_num][0]) +
                    (box[i][1] - box[(i + 1) % pts_num][1]) *
                            (box[i][1] - box[(i + 1) % pts_num][1]));
        }
        area = Math.abs(area / 2.0);
        return area * this.unclip_ratio / dist;
    }

    RotatedRect getMinAreaRect(MatOfPoint contour) {
        MatOfPoint2f point2f = new MatOfPoint2f(contour.toArray());
        RotatedRect rotatedRect = Imgproc.minAreaRect(point2f);
        point2f.release();
        return rotatedRect;
    }

    private double getMinEdgeLength(RotatedRect rotatedRect) {
        return Math.min(rotatedRect.size.width, rotatedRect.size.height);
    }

    private float[][] getMinAreaRectPoints(RotatedRect rotatedRect) {
        Mat points = new Mat();
        Imgproc.boxPoints(rotatedRect, points);
        float[] _boxPoints = new float[points.rows() * points.cols()];
        points.get(0, 0, _boxPoints);
        float[][] boxPoints = new float[4][2];
        for (int i = 0; i < boxPoints.length; i++) {
            System.arraycopy(_boxPoints, i * boxPoints[i].length,
                    boxPoints[i], 0, boxPoints[i].length);
        }
        points.release();
        return boxPoints;
    }

    static int clip(int v, int min, int max) {
        if (v < min) return min;
        if (v > max) return max;
        return v;
    }

    /**
     * 计数区域的平均概率
     */
    double boxScoreFast(Mat pred, RotatedRect miniBox) {
        Mat mask = Mat.zeros(pred.rows(), pred.cols(), CvType.CV_8U);
        float[][] pts = getMinAreaRectPoints(miniBox);
        MatOfPoint mop = new MatOfPoint();
        List<Point> lp = new ArrayList<>();
        for (int i = 0; i < pts.length; i++) {
            lp.add(new Point(pts[i][0], pts[i][1]));
        }
        mop.fromList(lp);

        Imgproc.fillPoly(mask, Arrays.asList(mop), new Scalar(1));
        Scalar mean = Core.mean(pred, mask);
        mop.release();
        mask.release();
        return mean.val[0];
    }

    public Bitmap getDetectedResult() {
        return detectedResult;
    }
}
