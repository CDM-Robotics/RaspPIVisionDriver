package team6072.vision;

import edu.wpi.first.networktables.NetworkTableInstance;
import edu.wpi.first.vision.*;
import org.opencv.core.RotatedRect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;

import edu.wpi.cscore.CvSource;
import edu.wpi.cscore.VideoSource;
import edu.wpi.first.cameraserver.CameraServer;
import edu.wpi.first.networktables.*;
import org.opencv.core.MatOfPoint2f;

import org.opencv.imgproc.Imgproc;

import org.opencv.core.Point;
import org.opencv.core.Rect;

import java.util.ArrayList;
import org.opencv.core.Mat;

import java.util.List;
import java.util.SortedMap;

import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;

public class CloseUpPipelineListener implements VisionRunner.Listener<CloseUpPipeline> {

    // The object to synchronize on to make sure the vision thread doesn't
    // write to variables the main thread is using.
    private final Object visionLock = new Object();

    // Network Table Entrys
    private NetworkTable mTbl;
    private NetworkTableEntry mX1;
    private NetworkTableEntry mY1;
    private NetworkTableEntry mX2;
    private NetworkTableEntry mY2;

    private final double TAPE_DIST_FROM_CENTER_INCHES_X = 5.65572;
    private final double TAPE_HEIGHT_IN_INCHES = 5.825352102;
    private final double CAMERA_FOV_ANGLE_X = 0.9098492;
    private final double CAMERA_FOV_ANGLE_Y = 1.028149899; // fix this bc its probably wrong
    private final int CAMERA_PIXEL_WIDTH_PIXELS = 160;
    private final int CAMERA_PIXEL_HEIGHT_PIXELS = 120;

    // Pipelines
    private CloseUpPipeline pipeline;

    // Camera Servers
    private CameraServer mCameraServer;
    private CvSource mHVSThresholdOutput;
    private CvSource mRectanglesOutput;

    // Counters
    private int mCallCounter = 0;
    private int mCounter;

    public CloseUpPipelineListener() {
        // instantiate Network Tables
        NetworkTableInstance tblInst = NetworkTableInstance.getDefault();
        mTbl = tblInst.getTable("vision");
        // Instantiate Camera Server Stuff
        mCameraServer = CameraServer.getInstance();
        // output HSVThreshold output to the camera server on Shuffleboard
        mHVSThresholdOutput = mCameraServer.putVideo("mHVSThresholdOutput", 320, 240);
        mRectanglesOutput = mCameraServer.putVideo("mRectanglesOutput", 320, 240);
    }

    public void outputColorFilters(CloseUpPipeline pipeline) {
        Mat erodeOutput = pipeline.hsvThresholdOutput();
        mHVSThresholdOutput.putFrame(erodeOutput);
    }

    public void outputRectangles(Mat mat) {
        mRectanglesOutput.putFrame(mat);
    }

    private boolean m_inCopyPipeline = false;

    /**
     * Called when the pipeline has run. We need to grab the output from the
     * pipeline then communicate to the rest of the system over network tables
     */
    @Override
    public void copyPipelineOutputs(CloseUpPipeline pipeline) {
        synchronized (visionLock) {
            if (m_inCopyPipeline) {
                return;
            }
            m_inCopyPipeline = true;
            mCallCounter++;
            // Manually makes sure that the program doesn't trip on itself
            // Take a snapshot of the pipeline's output because it may have changed the next
            // time this method is called!
            // Pipeline work and Camera Server processing
            // hsv output
            outputColorFilters(pipeline);
            // rectangles output
            ArrayList<MatOfPoint> mats = pipeline.findContoursOutput();
            ArrayList<RotatedRect> rotatedRects = new ArrayList<RotatedRect>();
            // Network Tables Stuff
            mTbl.getEntry("PIKey").setString("Call: " + mCounter);
            if (mats.size() != -1) {
                for (int i = 0; i < mats.size(); i++) {
                    RotatedRect rect = Imgproc.minAreaRect(new MatOfPoint2f(mats.get(i).toArray()));
                    rotatedRects.add(rect);
                }

                rotatedRects = orderFilterRectangles(rotatedRects);

                for (int i = 0; i < Math.min(2, rotatedRects.size()); i++) {
                    RotatedRect rect = rotatedRects.get(i);

                    Point center = rect.center;
                    double angle = rect.angle;
                    Size size = rect.size;

                    mTbl.getEntry("Rect " + i + " center X").setString("x = " + center.x);
                    mTbl.getEntry("Rect " + i + " center Y").setString("y = " + center.y);
                    mTbl.getEntry("Rect " + i + " angle").setDouble(angle);
                    mTbl.getEntry("Rect " + i + " Size").setString("Size : " + (size.width * size.height));
                }

                if (rotatedRects != null)
                    System.out.println("Hello");

                if (rotatedRects != null && rotatedRects.size() >= 2) {
                    mTbl.getEntry("Distance From Target X").setDouble(getDistanceFromTargetUsingXAxis(rotatedRects));
                    mTbl.getEntry("Distance From Tape 1 Inches")
                            .setDouble(getDistanceFromTargetUsingYAxis(rotatedRects.get(0)));
                    mTbl.getEntry("Distance From Tape 2 Inches")
                            .setDouble(getDistanceFromTargetUsingYAxis(rotatedRects.get(1)));
                }
            }
            mCounter++;
            m_inCopyPipeline = false;
        }
    }

    private class RectDistance {
        public double distance;
        public RotatedRect rect;
    }

    /**
     * Returns the same array list of 2 RotatedRectangles but ordered from left to
     * right - this is so that subsequent calculations so longer have to worry about
     * the ordering of the rectangles
     * 
     * @param rotatedRects - an array list of the RotatedRectangles directly from
     *                     the camera's contour output
     * @return - an array list of Rotated Rectangles
     */
    private ArrayList<RotatedRect> orderFilterRectangles(ArrayList<RotatedRect> rotatedRects) {
        if (rotatedRects.size() >= 2) {
            ArrayList<RectDistance> map = new ArrayList<RectDistance>(rotatedRects.size());
            for (RotatedRect curRect : rotatedRects) {
                double distFromCenterR1 = abs(curRect.center.x - (CAMERA_PIXEL_WIDTH_PIXELS / 2));
                RectDistance curRectDistance = new RectDistance();
                curRectDistance.distance = distFromCenterR1;
                curRectDistance.rect = curRect;

                if (map.size() == 0) {
                    map.add(curRectDistance);
                }
                for (int i = 0; i < map.size(); i++) {
                    RectDistance mapRectDistance = map.get(i);
                    if (curRectDistance.distance < mapRectDistance.distance) {
                        map.add(i, curRectDistance);
                    }
                }
            }

            RectDistance list0 = map.get(0);
            RectDistance list1 = map.get(1);

            if (list0.rect.center.x > list1.rect.center.x) {
                rotatedRects.set(0, list1.rect);
                rotatedRects.set(1, list0.rect);
            }
            else {
                rotatedRects.set(0, list0.rect);
                rotatedRects.set(1, list1.rect);
            }
            return rotatedRects;
        }
        return null;
    }

    /**
     * Calculates the center of mass between two objects of varying pixel mass
     */
    private double centerOfMass(double size1, double size2, double position1, double position2) {
        return (((size1 * position1) + (size2 * position2)) / (position1 + position2));
    }

    public double abs(double num) {
        if (num < 0) {
            num = num * -1;
        }
        return num;
    }

    /**
     * This function finds the distance between the camera and the target, -
     * assuming that the target is perfectly perpendicular to the target
     * horizantally - uses the horizantal axis ONLY, meaning it does not take into
     * account vertical distortion
     * 
     * @param rotatedRects - the Array list of ordered Rotated Rectanlges
     * @return - returns the distance inbetween the camera and the target
     */
    private double getDistanceFromTargetUsingXAxis(ArrayList<RotatedRect> rotatedRects) {

        // Calculate the center of mass of the tape
        double mass1 = rotatedRects.get(0).size.height * rotatedRects.get(0).size.width;
        double mass2 = rotatedRects.get(1).size.height * rotatedRects.get(1).size.width;

        double massCenterXpx = centerOfMass(mass1, mass2, rotatedRects.get(0).center.x, rotatedRects.get(1).center.x);
        double massCenterYpx = centerOfMass(mass1, mass2, rotatedRects.get(0).center.y, rotatedRects.get(1).center.y);

        // Calculate the distance from the target X horizontally
        double tapeDistFromCenterPxX = abs(rotatedRects.get(0).center.x - massCenterXpx);
        double halfOfCameraPixelWidthInches = (TAPE_DIST_FROM_CENTER_INCHES_X / tapeDistFromCenterPxX)
                * (CAMERA_PIXEL_WIDTH_PIXELS / 2);
        double distanceFromTargetX = halfOfCameraPixelWidthInches / (java.lang.Math.tan(CAMERA_FOV_ANGLE_X / 2));

        mTbl.getEntry("Center Of Mass X").setDouble(massCenterXpx);
        mTbl.getEntry("Center Of Mass Y").setDouble(massCenterYpx);

        return distanceFromTargetX;
    }

    /**
     * 
     * @param rect  - the desired rotated rectangle
     * @param angle - the angle the rectangle is tilted at
     * @return
     */
    private double findHeight(RotatedRect rect) {
        if ((rect.size.height) > (rect.size.width)) {
            return rect.size.height;
        } else {
            return rect.size.width;
        }
    }

    /**
     * Calculates the distance from target X horizantally - using the vertical Axis
     * - assuming it is perpendicular to the wall vertically
     * 
     * @param rotatedRects
     * @return
     */
    private double getDistanceFromTargetUsingYAxis(RotatedRect rect) {
        double tapeHeightPx = findHeight(rect);

        double angle = (tapeHeightPx / CAMERA_PIXEL_HEIGHT_PIXELS) * CAMERA_FOV_ANGLE_Y;
        double pxToInches = (TAPE_HEIGHT_IN_INCHES / tapeHeightPx);
        double distanceFromTapePx = (tapeHeightPx / 2) / java.lang.Math.tan(angle / 2);
        double distanceFromTapeInches = distanceFromTapePx * pxToInches;

        return distanceFromTapeInches;
    }
}