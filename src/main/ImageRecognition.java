package main;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;

import javax.imageio.ImageIO;

import org.lwjgl.BufferUtils;
import org.lwjgl.opengl.Display;
import org.lwjgl.opengl.GL11;
import org.lwjgl.util.vector.Vector3f;
import org.opencv.calib3d.Calib3d;
import org.opencv.core.Core;
import org.opencv.core.Core.MinMaxLocResult;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfDMatch;
import org.opencv.core.MatOfDouble;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.MatOfPoint3f;
import org.opencv.core.Point;
import org.opencv.core.Point3;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.features2d.DMatch;
import org.opencv.features2d.DescriptorExtractor;
import org.opencv.features2d.DescriptorMatcher;
import org.opencv.features2d.FeatureDetector;
import org.opencv.features2d.Features2d;
import org.opencv.features2d.KeyPoint;
import org.opencv.highgui.Highgui;
import org.opencv.highgui.VideoCapture;
import org.opencv.imgproc.Imgproc;

import engineTester.PoseEstimationValues;
import entities.Camera;
import entities.Entity;
import javafx.event.ActionEvent;
import javafx.fxml.FXML;
import javafx.scene.control.Button;
import javafx.scene.control.Slider;
import javafx.scene.control.TextField;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;
import main.utils.Utils;
import models.RawModel;
import models.TexturedModel;
import renderEngine.DisplayManager;
import renderEngine.Loader;
import renderEngine.Renderer;
import shaders.StaticShader;
import textures.ModelTexture;

/**
 * This class implements the main image recognition steps needed for the object/cube recognition approach.
 * In particular template matching, ORB and pose estimation via solvePnP OpenCV functions are used.
 * 
 */
public class ImageRecognition
{	
	private static final String PATH_IMAGES = "images/";
	
	private static final Scalar COLOR_BLUE = new Scalar(255, 200, 50);
	private static final Scalar COLOR_GREEN = new Scalar(0, 255, 0);
	private static final Scalar COLOR_RED = new Scalar(0, 0, 255);
	
	private static int cameraId = 0;
	
	// the FXML button
	@FXML
	private Button button;
	// the FXML image view
	@FXML
	private ImageView currentFrame;
	
	// a timer for acquiring the video stream
	private ScheduledExecutorService timer;
	// the OpenCV object that realizes the video capture
	private VideoCapture capture = new VideoCapture();
	// a flag to change the button behavior
	private boolean cameraActive = false;
	// the id of the camera to be used
	
	private static Mat img1 = Highgui.imread(PATH_IMAGES + "5house.png");
	private static Mat img2 = Highgui.imread(PATH_IMAGES + "6lena.png");
	private static Mat img3 = Highgui.imread(PATH_IMAGES + "black.png");

	private static Mat imgCurrent;	
	private String textImage = "";	
	private String textImageName;	
	private String textTranslation;
	private String textRotation;
	private int rotX, rotY, rotZ;
	private int transX, transY, transZ;
	
	private float nndrRatio = 1;
	private boolean drawMatches = false;
	private boolean drawProjectedCube = false;
	@FXML
    private Slider mySlider;
    @FXML
    private TextField textField;
	
	private MatOfPoint3f objectPoints3d;
	private MatOfPoint2f scenePoints2d;
	
	private Mat homography;
	
	private static boolean startPreview = true;
	private int tCount = 0;
	private static boolean printOut = true;	
	
	/**
	 * Get a frame from the opened video stream (if any)
	 *
	 * @return the {@link Mat} to show
	 */
	private Mat grabFrame()
	{
		// init everything
		Mat img = new Mat();
		
		// check if the capture is open
		if (this.capture.isOpened())
		{
			try
			{				
				this.capture.read(img); // read the current frame
				matchTemplate(img);
				matchFeaturePoints(img, objectPoints3d, scenePoints2d);
				drawInfosOnFrame(img);	
			}
			catch (Exception e)
			{
				// log the error
				System.err.println("Exception during the image elaboration: " + e);
			}
		}
		
		return img;
	}

    public void initialize() {
    	nndrRatio = 0.75f;
    	mySlider.setValue(nndrRatio);

        mySlider.valueProperty().addListener((observable, oldValue, newValue) -> {
            nndrRatio = ((Double) newValue).floatValue();            
        });
    }
    
	public static void testOpenGL() {

		DisplayManager.createDisplay();

		Loader loader = new Loader();
		StaticShader shader = new StaticShader();
		Renderer renderer = new Renderer(shader);

		float[] vertices = {			
				-0.5f,0.5f,-0.5f,	
				-0.5f,-0.5f,-0.5f,	
				0.5f,-0.5f,-0.5f,	
				0.5f,0.5f,-0.5f,		
				
				-0.5f,0.5f,0.5f,	
				-0.5f,-0.5f,0.5f,	
				0.5f,-0.5f,0.5f,	
				0.5f,0.5f,0.5f,
				
				0.5f,0.5f,-0.5f,	
				0.5f,-0.5f,-0.5f,	
				0.5f,-0.5f,0.5f,	
				0.5f,0.5f,0.5f,
				
				-0.5f,0.5f,-0.5f,	
				-0.5f,-0.5f,-0.5f,	
				-0.5f,-0.5f,0.5f,	
				-0.5f,0.5f,0.5f,
				
				-0.5f,0.5f,0.5f,
				-0.5f,0.5f,-0.5f,
				0.5f,0.5f,-0.5f,
				0.5f,0.5f,0.5f,
				
				-0.5f,-0.5f,0.5f,
				-0.5f,-0.5f,-0.5f,
				0.5f,-0.5f,-0.5f,
				0.5f,-0.5f,0.5f				
		};
		
		int[] indices = {
				0,1,3,	
				3,1,2,	
				
				4,5,7,
				7,5,6,
				
				8,9,11,
				11,9,10,
				
				12,13,15,
				15,13,14,
				
				16,17,19,
				19,17,18,
				
				20,21,23,
				23,21,22
		};
		
		float[] textureCoords = {
				//back
				1,0.5f,
				1,1,
				0.5f,1,
				0.5f,0.5f,
				//front
				0.5f,0,
				0.5f,0.5f,
				1,0.5f,
				1,0,	
				//right
				0.5f,0.5f,
				0.5f,1,	
				0,1,
				0,0.5f,			
				//left
				0,0,
				0,0.5f,
				0.5f,0.5f,
				0.5f,0,
				//above
				0,0.5f,
				0,0,
				0.5f,0,
				0.5f,0.5f,
				//below
				0,0,
				0,1,
				1,1,
				1,0			
		};

		RawModel model = loader.loadToVAO(vertices, textureCoords, indices);
		//RawModel model = OBJLoader.loadObjModel("cube3D", loader);
		TexturedModel staticModel = new TexturedModel(model,new ModelTexture(loader.loadTexture("cube3D")));
		Entity entity = new Entity(staticModel, new Vector3f(0,0,-2.5f),0,0,0,1);
		Camera camera = new Camera();
		
		while (!Display.isCloseRequested()) {

			if(imgCurrent == img1) {
				
				entity.setRotX(PoseEstimationValues.rotX);
				entity.setRotY(PoseEstimationValues.rotY);
				
				camera.roll(PoseEstimationValues.rotZ);
			} else {
				entity.setPosition(new Vector3f(0, 0, -2.5f));
				
				entity.setRotX(PoseEstimationValues.rotX);
				entity.setRotY(90+PoseEstimationValues.rotY);
				
				camera.roll(PoseEstimationValues.rotZ);
			}
			
			camera.move();
			renderer.prepare();
			shader.start();
			shader.loadViewMatrix(camera);
			renderer.render(entity,shader);
			shader.stop();
			DisplayManager.updateDisplay();
		}

		shader.cleanUp();
		loader.cleanUp();
		DisplayManager.closeDisplay();
		
		startPreview = true;
	}
	
	private void drawInfosOnFrame(Mat img) {	
		Core.putText(img, textImage, new Point(50,50), 3, 1, COLOR_BLUE, 2);
		
		textRotation = "Rotation:   ";
		textRotation += " x=";
		textRotation += rotX;
		textRotation += " y=";
		textRotation += rotY;
		textRotation += " z=";
		textRotation += rotZ;
		
		Core.putText(img, textRotation, new Point(50,450), 3, 1, COLOR_RED, 2);
		
		textTranslation = "Translation:";
		textTranslation += " x=";
		textTranslation += transX;
		textTranslation += " y=";
		textTranslation += transY;
		textTranslation += " z=";
		textTranslation += transZ;
		Core.putText(img, textTranslation, new Point(50,400), 3, 1, COLOR_RED, 2);
	}
	
	/**
	 * The action triggered by pushing the button on the GUI
	 *
	 * @param event
	 *            the push button event
	 * @return 
	 */
	@FXML
	protected void start3dModelPreview(ActionEvent event)
	{		
		if(startPreview) {
			startPreview = false;
			testOpenGL();
		}		
	}
	
	@FXML
	protected void drawMatchesSwitch(ActionEvent event)
	{		
		drawMatches = ! drawMatches;
	}
	
	@FXML
	protected void drawProjectedCube(ActionEvent event)
	{		
		drawProjectedCube = ! drawProjectedCube;
	}
	
	@FXML
	protected void test(ActionEvent event)
	{		
		GL11.glReadBuffer(GL11.GL_FRONT);
		int width = Display.getDisplayMode().getWidth();
		int height= Display.getDisplayMode().getHeight();
		int bpp = 4; // Assuming a 32-bit display with a byte each for red, green, blue, and alpha.
		ByteBuffer buffer = BufferUtils.createByteBuffer(width * height * bpp);
		GL11.glReadPixels(0, 0, width, height, GL11.GL_RGBA, GL11.GL_UNSIGNED_BYTE, buffer );
		
		File file = new File("C:/Users/Marius/Desktop/test123.PNG"); // The file to save to.
		String format = "PNG"; // Example: "PNG" or "JPG"
		BufferedImage image = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
		   
		for(int x = 0; x < width; x++) 
		{
		    for(int y = 0; y < height; y++)
		    {
		        int i = (x + (width * y)) * bpp;
		        int r = buffer.get(i) & 0xFF;
		        int g = buffer.get(i + 1) & 0xFF;
		        int b = buffer.get(i + 2) & 0xFF;
		        image.setRGB(x, height - (y + 1), (0xFF << 24) | (r << 16) | (g << 8) | b);
		    }
		}
		   
		try {
		    ImageIO.write(image, format, file);
		} catch (IOException e) { e.printStackTrace(); }
	}
	
	/**
	 * The action triggered by pushing the button on the GUI
	 *
	 * @param event
	 *            the push button event
	 */
	@FXML
	protected void startCamera(ActionEvent event)
	{			
		if (!this.cameraActive)
		{
			// start the video capture
			this.capture.open(cameraId);
			
			// is the video stream available?
			if (this.capture.isOpened())
			{
				this.cameraActive = true;
				
				// grab a frame every 33 ms (30 frames/sec)
				Runnable frameGrabber = new Runnable() {
					
					@Override
					public void run()
					{
						// effectively grab and process a single frame
						Mat frame = grabFrame();
						
						// convert and show the frame
						Image imageToShow = Utils.mat2Image(frame);
						updateImageView(currentFrame, imageToShow);
					}
				};
				
				this.timer = Executors.newSingleThreadScheduledExecutor();
				this.timer.scheduleAtFixedRate(frameGrabber, 0, 33, TimeUnit.MILLISECONDS);
				
				// update the button content
				this.button.setText("Stop Camera");
			}
			else
			{
				// log the error
				System.err.println("Impossible to open the camera connection...");
			}
		}
		else
		{
			
			// the camera is not active at this point
			this.cameraActive = false;
			// update again the button content
			this.button.setText("Start Camera");
			
			// stop the timer
			this.stopAcquisition();
			
			//testOpenGL();
		}
	}
	
	@SuppressWarnings("unused")
	private void chessBoardAR(Mat img) {
		Size chessSize = new Size(9,6);
		MatOfPoint2f corners = new MatOfPoint2f();
		boolean foundChessboard = Calib3d.findChessboardCorners(img, chessSize, corners);
		Calib3d.drawChessboardCorners(img, chessSize, corners, foundChessboard);
		
		float BoardBoxSize=3;
		List<Point3> listCorners3d = new ArrayList<Point3>();
		for (int j = 0; j < chessSize.height; j++) {
			for (int i = 0; i < chessSize.width; i++) {
				listCorners3d.add(new Point3(i * BoardBoxSize, j * BoardBoxSize, 0));
			}
		}
		MatOfPoint3f corners3d = new MatOfPoint3f();
		corners3d.fromList(listCorners3d);
		
		// Camera internals
	    double focal_length = imgCurrent.cols(); // Approximate focal length.
	    Point center = new Point(imgCurrent.cols()/2,imgCurrent.rows()/2);
		
		Mat cameraMatrix = new Mat(3, 3, CvType.CV_64FC1);
		int row = 0, col = 0;
		cameraMatrix.put(row ,col, focal_length, 0, center.x, 0, focal_length, center.y, 0, 0, 1 );

		Mat rvec = new Mat();
		Mat tvec = new Mat();
	
		MatOfPoint3f objectPoints = new MatOfPoint3f();
		List<Point3> listObjectPoints = new ArrayList<Point3>();
	  //listObjectPoints.add(new Point3(  0,  0,  0));
		listObjectPoints.add(new Point3(  0,  0,-10));
		listObjectPoints.add(new Point3( 10,  0,  0));
		listObjectPoints.add(new Point3(  0, 10,  0));
		listObjectPoints.add(new Point3( 10, 10,  0));
		listObjectPoints.add(new Point3( 10, 10,-10));
		listObjectPoints.add(new Point3(  0, 10,-10));
		listObjectPoints.add(new Point3(  10, 0,-10));
		
		listObjectPoints.add(new Point3(33, 0,-1.0));//(x,y,z)
		listObjectPoints.add(new Point3(12, 8,-1.0));
		listObjectPoints.add(new Point3(20, 8,-1.0));
		listObjectPoints.add(new Point3(20, 0,-1.0));
		
		objectPoints.fromList(listObjectPoints);
		MatOfDouble distCoeffs = new MatOfDouble();
		MatOfPoint2f imagePoints = new MatOfPoint2f();
		Calib3d.projectPoints(objectPoints, rvec, tvec, cameraMatrix, distCoeffs, imagePoints);

		List<Point> listCorners = corners.toList();
		List<Point> listPoints = imagePoints.toList();
		
		Core.line(img, listCorners.get(0), listPoints.get(0), new Scalar(0, 255, 0), 3);
		Core.line(img, listCorners.get(0), listPoints.get(1), new Scalar(255, 0, 0), 3);
		Core.line(img, listCorners.get(0), listPoints.get(2), new Scalar(0, 0, 255), 3);
		Core.line(img, listPoints.get(2), listPoints.get(3), new Scalar(0, 255, 255), 3);	
		Core.line(img, listPoints.get(3), listPoints.get(1), new Scalar(0, 255, 255), 3);	
		Core.line(img, listPoints.get(3), listPoints.get(4), new Scalar(0, 255, 255), 3);
		Core.line(img, listPoints.get(2), listPoints.get(5), new Scalar(0, 255, 255), 3);
		Core.line(img, listPoints.get(1), listPoints.get(6), new Scalar(0, 255, 255), 3);	

		Core.line(img, listPoints.get(0), listPoints.get(5), new Scalar(0, 255, 255), 3);
		Core.line(img, listPoints.get(0), listPoints.get(6), new Scalar(0, 255, 255), 3);	
		Core.line(img, listPoints.get(4), listPoints.get(5), new Scalar(0, 255, 255), 3);	
		Core.line(img, listPoints.get(4), listPoints.get(6), new Scalar(0, 255, 255), 3);	
	}
	
	private void matchTemplate(Mat img) {
		int match_method = Imgproc.TM_SQDIFF; //Imgproc.TM_CCOEFF;// 
		
		// Create the result matrix
	    int result_cols = img.cols() - img1.cols() + 1;
	    int result_rows = img.rows() - img1.rows() + 1;
	    Mat result = new Mat(result_rows, result_cols, CvType.CV_32FC1);

	    // Do the Matching
	    Imgproc.matchTemplate(img, img1, result, match_method);

	    // Localizing the best match with minMaxLoc
	    MinMaxLocResult mmr = Core.minMaxLoc(result);
	    
	    double min1 = mmr.minVal;
	    
	    int result_cols2 = img.cols() - img2.cols() + 1;
	    int result_rows2 = img.rows() - img2.rows() + 1;
	    Mat result2 = new Mat(result_rows2, result_cols2, CvType.CV_32FC1);
	    Imgproc.matchTemplate(img, img2, result2, match_method);
	    
	    MinMaxLocResult mmr2 = Core.minMaxLoc(result2);
	    
	    double min2 = mmr2.minVal;
	    
	    int result_cols3 = img.cols() - img3.cols() + 1;
	    int result_rows3 = img.rows() - img3.rows() + 1;
	    Mat result3 = new Mat(result_rows3, result_cols3, CvType.CV_32FC1);
	    Imgproc.matchTemplate(img, img3, result3, match_method);
	    
	    MinMaxLocResult mmr3 = Core.minMaxLoc(result3);
	    
	    double min3 = mmr3.minVal;			    

	    if (min1 < min2 && min1 < min3 ) {
			textImageName = " (Bild 1)";
			imgCurrent = img1;	    	
		} else if (min2 < min3 && min2 < min1) {
			textImageName = " (Bild 2)";
			imgCurrent = img2;
		}	else if (min3 < min1 && min3 < min2) {
			textImageName = " (Bild 3)";
			imgCurrent = img3;
		}	    
	}
	
	private void matchFeaturePoints(Mat img, MatOfPoint3f objectPoints3d, MatOfPoint2f scenePoints2d) {
		
		FeatureDetector featureDetector = FeatureDetector.create(FeatureDetector.ORB);	
        DescriptorExtractor descriptorExtractor = DescriptorExtractor.create(DescriptorExtractor.ORB);
        DescriptorMatcher descriptorMatcher = DescriptorMatcher.create(DescriptorMatcher.BRUTEFORCE_HAMMING);

		// get SOURCE image keypoints from source image
		MatOfKeyPoint objectKeyPoints = new MatOfKeyPoint();			
        featureDetector.detect(imgCurrent, objectKeyPoints);
        
        MatOfKeyPoint objectDescriptors = new MatOfKeyPoint();
        descriptorExtractor.compute(imgCurrent, objectKeyPoints, objectDescriptors);
        
        // get SCENE image keypoints
        MatOfKeyPoint sceneKeyPoints = new MatOfKeyPoint();
        featureDetector.detect(img, sceneKeyPoints);

        // Computing descriptors in background image
        MatOfKeyPoint sceneDescriptors = new MatOfKeyPoint();
        descriptorExtractor.compute(img, sceneKeyPoints, sceneDescriptors); 
        
        List<MatOfDMatch> matches = new LinkedList<MatOfDMatch>();        
        descriptorMatcher.knnMatch(objectDescriptors, sceneDescriptors, matches, 2); // matching object and scene images        

        MatOfDMatch goodMatches = new MatOfDMatch();
        LinkedList<DMatch> goodMatchesList = new LinkedList<DMatch>();        
        for (int i = 0; i < matches.size(); i++) {
            MatOfDMatch matofDMatch = matches.get(i);
            DMatch[] dmatcharray = matofDMatch.toArray();            
            DMatch m1 = dmatcharray[0];
            DMatch m2 = dmatcharray[1];

            if (m1.distance <= m2.distance * nndrRatio) {
                goodMatchesList.addLast(m1);
            }
        }

        if (goodMatchesList.size() >= 10) {
	    	textImage = "Objekt erkannt!";
	    	textImage += textImageName;

            List<KeyPoint> objKeypointlist = objectKeyPoints.toList();
            List<KeyPoint> scnKeypointlist = sceneKeyPoints.toList();

            LinkedList<Point> objectPoints = new LinkedList<>();
            LinkedList<Point> scenePoints = new LinkedList<>();             

            Point3 arrayPoints3D[] = new Point3[goodMatchesList.size()];
            Point arrayPoints2D[] = new Point[goodMatchesList.size()];
            
            MatOfPoint3f objPoints = new MatOfPoint3f();
            MatOfPoint2f imgPoints = new MatOfPoint2f();            

            for (int i = 0; i < goodMatchesList.size(); i++) {
            	Point oPoint = objKeypointlist.get(goodMatchesList.get(i).queryIdx).pt;
                objectPoints.addLast(oPoint);
            	Point sPoint = scnKeypointlist.get(goodMatchesList.get(i).trainIdx).pt;;
                scenePoints.addLast(sPoint);
                
                arrayPoints3D[i]= new Point3(sPoint.x, sPoint.y, 0);
                arrayPoints2D[i] = new Point(oPoint.x, oPoint.y);
            }

            objPoints.fromArray(arrayPoints3D);
            imgPoints.fromArray(arrayPoints2D);
            
            objectPoints3d = objPoints;
            scenePoints2d = imgPoints;

            MatOfPoint2f objMatOfPoint2f = new MatOfPoint2f();
            objMatOfPoint2f.fromList(objectPoints);
            MatOfPoint2f scnMatOfPoint2f = new MatOfPoint2f();
            scnMatOfPoint2f.fromList(scenePoints);

            homography = Calib3d.findHomography(objMatOfPoint2f, scnMatOfPoint2f, Calib3d.RANSAC, 5);  

            Mat obj_corners = new Mat(4, 1, CvType.CV_32FC2);
            Mat scene_corners = new Mat(4, 1, CvType.CV_32FC2);

            obj_corners.put(0, 0, new double[]{0, 0});
            obj_corners.put(1, 0, new double[]{imgCurrent.cols(), 0});
            obj_corners.put(2, 0, new double[]{imgCurrent.cols(), imgCurrent.rows()});
            obj_corners.put(3, 0, new double[]{0, imgCurrent.rows()});

            // Transforming object corners to scene corners
            Core.perspectiveTransform(obj_corners, scene_corners, homography);
            
            Point3 leftUp3d =    new Point3(0,0,0);
            Point3 rightUp3d =   new Point3(1,0,0);
            Point3 rightDown3d = new Point3(1,1,0);
            Point3 leftDown3d =  new Point3(0,1,0);
            
            Point leftUp = new Point(scene_corners.get(0, 0));
            Point rightUp = new Point(scene_corners.get(1, 0));
            Point rightDown = new Point(scene_corners.get(2, 0));
            Point leftDown = new Point(scene_corners.get(3, 0));
            
            Core.line(img, leftUp, rightUp, COLOR_GREEN, 4);
            Core.line(img, rightUp, rightDown, COLOR_GREEN, 4);
            Core.line(img, rightDown, leftDown, COLOR_GREEN, 4);
            Core.line(img, leftDown, leftUp, COLOR_GREEN, 4);
            
    		Core.circle(img, leftUp, 10, COLOR_RED, 3);
    		Core.circle(img, rightUp, 10, COLOR_RED, 3);
    		Core.circle(img, leftDown, 10, COLOR_RED, 3);
    		Core.circle(img, rightDown, 10, COLOR_RED, 3);
            
            Mat cameraMatrix = new Mat(3, 3, CvType.CV_64FC1);
    		cameraMatrix.put(0 ,0, imgCurrent.cols(), 0, imgCurrent.cols()/2, 0, imgCurrent.cols(), imgCurrent.rows()/2, 0, 0, 1 );
    		
    		MatOfPoint3f objPointsTest = new MatOfPoint3f(leftUp3d, rightUp3d, rightDown3d, leftDown3d);
            MatOfPoint2f imgPointsTest = new MatOfPoint2f(leftUp, rightUp, rightDown, leftDown); 
    		
    		Mat rvec = new Mat();
    		Mat tvec = new Mat();
    		Calib3d.solvePnP(objPointsTest, imgPointsTest, cameraMatrix, new MatOfDouble(), rvec, tvec); // perform pose estimation
    		//Calib3d.solvePnP(objPointsTest, imgPointsTest, cameraMatrix, new MatOfDouble(), rvec, tvec, cameraActive, Calib3d.CV_P3P);
    		
    		if(drawProjectedCube) {
        		
        		MatOfPoint3f objectPointsCube = new MatOfPoint3f();
        		List<Point3> listObjectPoints = new ArrayList<Point3>();
        		double height = 0.25;
        	  //listObjectPoints.add(new Point3(  0, 0, 0)); // origin
        		listObjectPoints.add(new Point3(  0, 0,height));
        		listObjectPoints.add(new Point3(  1, 0, 0));
        		listObjectPoints.add(new Point3(  0, 1, 0));
        		listObjectPoints.add(new Point3(  1, 1, 0));
        		listObjectPoints.add(new Point3(  1, 1,height));
        		listObjectPoints.add(new Point3(  0, 1,height));
        		listObjectPoints.add(new Point3(  1, 0,height));

        		objectPointsCube.fromList(listObjectPoints);
        		MatOfDouble distCoeffs = new MatOfDouble();
        		MatOfPoint2f imagePoints = new MatOfPoint2f();
        		Calib3d.projectPoints(objectPointsCube, rvec, tvec, cameraMatrix, distCoeffs, imagePoints);

        		Point origin = leftUp;
        		List<Point> listPoints = imagePoints.toList();
        		Core.line(img, origin, listPoints.get(0), new Scalar(0, 255, 255), 3);
        		Core.line(img, origin, listPoints.get(1), new Scalar(0, 255, 255), 3);
        		Core.line(img, origin, listPoints.get(2), new Scalar(0, 255, 255), 3);
        		Core.line(img, listPoints.get(2), listPoints.get(3), new Scalar(0, 255, 255), 3);	
        		Core.line(img, listPoints.get(3), listPoints.get(1), new Scalar(0, 255, 255), 3);	
        		Core.line(img, listPoints.get(3), listPoints.get(4), new Scalar(0, 255, 255), 3);
        		Core.line(img, listPoints.get(2), listPoints.get(5), new Scalar(0, 255, 255), 3);
        		Core.line(img, listPoints.get(1), listPoints.get(6), new Scalar(0, 255, 255), 3);	

        		Core.line(img, listPoints.get(0), listPoints.get(5), new Scalar(0, 255, 255), 3);
        		Core.line(img, listPoints.get(0), listPoints.get(6), new Scalar(0, 255, 255), 3);	
        		Core.line(img, listPoints.get(4), listPoints.get(5), new Scalar(0, 255, 255), 3);	
        		Core.line(img, listPoints.get(4), listPoints.get(6), new Scalar(0, 255, 255), 3);    			
    		}	
            
    		Size sizeTrans = tvec.size();
    		for (int i = 0; i < sizeTrans.height; i++) {
    		    for (int j = 0; j < sizeTrans.width; j++) { // j always 0: size.width == 0
    		        double[] data = tvec.get(i, j);
    		        for(double x : data) {
    		        	switch(i) {
    	        		case 0:
    	        			transX = (int) (10*x) -3;
    	        			PoseEstimationValues.posX = transX;
    	        			PoseEstimationValues.posX /= 5;
    	        			break;
    	        		case 1:
    	        			transY = (int) (10*x);
    	        			PoseEstimationValues.posY = transY;
    	        			PoseEstimationValues.posY /= 5;
    	        			break;
    	        		case 2:
    	        			transZ = (int) (10*x)-10;
    	        			PoseEstimationValues.posZ = transZ;
    	        			PoseEstimationValues.posZ /= 5;
    	        			break;
    	        		default:	        	
    		        	}
    		        }
    		    }
    		}

            Size size = rvec.size();
    		for (int i = 0; i < size.height; i++) {
    		    for (int j = 0; j < size.width; j++) { // j always 0: size.width == 0
    		        double[] data = rvec.get(i, j);
    		        for(double x : data) {
    		        	double degree = 180 * x / Math.PI;
    		        	switch(i) {
    	        		case 0:
    	        			rotX = (int) Math.round(degree);
    	        			PoseEstimationValues.rotX = rotX;
    	        			break;
    	        		case 1:
    	        			rotY = (int) Math.round(degree);
    	        			PoseEstimationValues.rotY = rotY;
    	        			break;
    	        		case 2:
    	        			rotZ = (int) Math.round(degree);
    	        			PoseEstimationValues.rotZ = -rotZ;
    	        			break;
    	        		default:	        	
    		        	}
    		        }
    		    }
    		}
            goodMatches.fromList(goodMatchesList); 
            
        } else {
	    	imgCurrent = img3;
	    	textImage = "Bitte Objekt fixieren...";	  
	    	rotX = 0;
	    	rotY = 0;
	    	rotZ = 0;
	    	transX = 0;
	    	transY = 0;
	    	transZ = 0;
        }   
        
        if(drawMatches) {
        	Features2d.drawMatches(imgCurrent, objectKeyPoints, img.clone(), sceneKeyPoints, goodMatches, img);
        }    
	}
	
	@SuppressWarnings("unused")
	private void testPoseEstimation(MatOfPoint3f objPoints, MatOfPoint2f imgPoints) {

		tCount++;
		if(tCount < 7) {
			return;
		}
		tCount = 0;
		
		// Camera internals
	    double focal_length = imgCurrent.cols(); // Approximate focal length.
	    Point center = new Point(imgCurrent.cols()/2,imgCurrent.rows()/2);
		
		Mat cameraMatrix = new Mat(3, 3, CvType.CV_64FC1);
		int row = 0, col = 0;
		cameraMatrix.put(row ,col, focal_length, 0, center.x, 0, focal_length, center.y, 0, 0, 1 );
		
		Mat rvec = new Mat();
		Mat tvec = new Mat();
		
		MatOfDouble mRMat = new MatOfDouble(3, 3, CvType.CV_32F);
		
		if (objPoints != null && imgPoints != null) {
			Calib3d.solvePnP(objPoints, imgPoints, cameraMatrix, new MatOfDouble(), rvec, tvec);
			//Calib3d.projectPoints(objectPoints, rvec, tvec, cameraMatrix, distCoeffs, imagePoints);
		} else {
			return;
		}
		
		Size size = rvec.size();
		for (int i = 0; i < size.height; i++) {
		    for (int j = 0; j < size.width; j++) { // j always 0: size.width == 0
		        double[] data = rvec.get(i, j);
		        for(double x : data) {
		        	double degree = 180 * x / Math.PI;
		        	switch(i) {
	        		case 0:
	        			rotX = (int) Math.round(degree);
	        			System.out.println("X: " + Math.round(degree));
	        			break;
	        		case 1:
	        			rotY = (int) Math.round(degree);
	        			System.out.println("Y: " + Math.round(degree));
	        			break;
	        		case 2:
	        			rotZ = (int) Math.round(degree);
	        			System.out.println("Z: " + rotZ);
	        			break;
	        		default:	        	
		        	}
		        }
		    }
		}
		
		Mat dst = new Mat();
		Calib3d.Rodrigues(rvec, dst);
		
		Mat projMatrix = new Mat(3,4,CvType.CV_64FC1);	
		projMatrix.put(0, 0, dst.get(0, 0));
		projMatrix.put(0, 1, dst.get(0, 1));
		projMatrix.put(0, 2, dst.get(0, 2));
		projMatrix.put(0, 3, 0);
		projMatrix.put(1, 0, dst.get(1, 0));
		projMatrix.put(1, 1, dst.get(1, 1));
		projMatrix.put(1, 2, dst.get(1, 2));
		projMatrix.put(1, 3, 0);
		projMatrix.put(2, 0, dst.get(2, 0));
		projMatrix.put(2, 1, dst.get(2, 1));
		projMatrix.put(2, 2, dst.get(2, 2));
		projMatrix.put(2, 3, 0);
		
		Mat rotMatrix = new Mat();
		Mat transVect = new Mat();
		Mat rotMatrixX = new Mat();
		Mat rotMatrixY = new Mat();
		Mat rotMatrixZ = new Mat();
		
		Mat eulerAngles = new Mat();		
		Calib3d.decomposeProjectionMatrix(projMatrix, cameraMatrix, rotMatrix, transVect, rotMatrixX, rotMatrixY, rotMatrixZ, eulerAngles);
		
		System.out.println(eulerAngles.dump());
		out("----------");
	}	
	
	/**
	 * Stop the acquisition from the camera and release all the resources
	 */
	private void stopAcquisition()
	{
		if (this.timer!=null && !this.timer.isShutdown())
		{
			try
			{
				// stop the timer
				this.timer.shutdown();
				this.timer.awaitTermination(33, TimeUnit.MILLISECONDS);
			}
			catch (InterruptedException e)
			{
				// log any exception
				System.err.println("Exception in stopping the frame capture, trying to release the camera now... " + e);
			}
		}
		
		if (this.capture.isOpened())
		{
			// release the camera
			this.capture.release();
		}
	}
	
	/**
	 * Update the {@link ImageView} in the JavaFX main thread
	 * 
	 * @param view
	 *            the {@link ImageView} to update
	 * @param image
	 *            the {@link Image} to show
	 */
	private void updateImageView(ImageView view, Image image)
	{
		Utils.onFXThread(view.imageProperty(), image);
	}
	
	/**
	 * On application close, stop the acquisition from the camera
	 */
	protected void setClosed()
	{
		this.stopAcquisition();
	}
	
	public static void printKeyPoints(MatOfKeyPoint keyPoints) {
        KeyPoint[] keypoints = keyPoints.toArray();
        
        int count = 0;
        for(KeyPoint keypoint : keypoints) {
            System.out.println(keypoint.toString());   
            count++;
        }
        System.out.println("Number of keypoints: " + count); 
	}

	private static void out(String text) {
		if(printOut) {
			System.out.println(text);
		}
	}
	
}
