#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/aruco.hpp>
#include <iostream>
#include <sstream>
#include <time.h>
#include <stdio.h>

#ifdef __APPLE__
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#endif

cv::Ptr<cv::aruco::Dictionary> dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);
cv::Mat camera_matrix;
cv::Mat dist_coeffs;

static void calcBoardCornerPositions(cv::Size boardSize, float squareSize, std::vector<cv::Point3f>& corners, std::string patternType){
    corners.clear();

    if (patternType == "CHESSBOARD" || patternType == "CIRCLES_GRID"){
			for( int i = 0; i < boardSize.height; ++i )
					for( int j = 0; j < boardSize.width; ++j )
							corners.push_back(cv::Point3f(float( j*squareSize ), float( i*squareSize ), 0));
		}

    else if (patternType == "ASYMMETRIC_CIRCLES_GRID"){
			for( int i = 0; i < boardSize.height; i++ )
					for( int j = 0; j < boardSize.width; j++ )
							corners.push_back(cv::Point3f(float((2*j + i % 2)*squareSize), float(i*squareSize), 0));
		}

}

//Future work: Add refinement to initial corner detection and post-calibration rms error
void calibrateCamera(cv::Mat& camera_matrix, cv::Mat& dist_coeffs, bool show_board_image){
	
	// Functional Variables
	std::vector<std::vector<cv::Point2f> > image_points;
	std::vector<std::vector<cv::Point3f> > object_points(1);
	cv::Size board_size;
	cv::Size calibration_image_size;
	std::vector<cv::Mat> rvecs;
	std::vector<cv::Mat> tvecs;
	
	// Contextual Parameters
	board_size.width = 9;
	board_size.height = 6;
	calibration_image_size.width = 720;
	calibration_image_size.height = 480;
	float calibration_square_size = 0.022;
	std::string calibration_pattern = "CHESSBOARD"; // Available Pattern Types: "CHESSBOARD", "CIRCLES_GRID", "ASYMMETRIC_CIRCLES_GRID"
	
	// NOTE: Maximum of 10 calibration images
	for(int i = 0 ; i < 11 ; i++){
		
		// STEP 1: Load Calibration Image
		std::stringstream convert;
		convert << i;
		std::string s;
		s = convert.str();
		cv::Mat board_image;
		board_image = cv::imread("calibration/" + s + ".jpg", CV_LOAD_IMAGE_COLOR);   
		
		if (! board_image.data){
			std::cout <<  "Total Number of calibration images: " << i << std::endl ;
			break;
		}
		
		// STEP 2: Search for calibration image keypoints
		std::vector<cv::Point2f> point_buf;
		bool found;
		cv::Mat gray;
		cv::cvtColor(board_image, gray, cv::COLOR_BGR2GRAY);
		found = cv::findChessboardCorners(gray, board_size, point_buf);
		if (found){			
			if (show_board_image){
				cv::drawChessboardCorners(board_image, board_size, cv::Mat(point_buf), found );
				cv::namedWindow( "Display window", cv::WINDOW_AUTOSIZE );
				cv::imshow( "Display window", board_image );
				cv::waitKey(0);
			}
			image_points.push_back(point_buf);
		}
		else{
			std::cout << "Could not detect key points for frame: " << i << std::endl;
		}

	}
	
	std::cout << "Number of patterns found: " << image_points.size() << std::endl;
	
	// STEP 3: Perform calibration if at least 5 good images are available
	if (image_points.size() >= 5){
		calcBoardCornerPositions(board_size, calibration_square_size, object_points[0], calibration_pattern);
		object_points.resize(image_points.size(),object_points[0]);
		double rms = calibrateCamera(object_points, image_points, calibration_image_size, camera_matrix, dist_coeffs, rvecs, tvecs);
	}
	
}

void arucoPoseEstimation(cv::Mat& input_image, int id, cv::Mat& tvec, cv::Mat& rvec, cv::Mat& mtx, cv::Mat& dist){
	// Contextual Parameters
	float aruco_square_size = 0.082;
	
	std::vector< int > marker_ids;
	std::vector< std::vector<cv::Point2f> > marker_corners, rejected_candidates;
	cv::Mat gray;
	
	cv::cvtColor(input_image, gray, cv::COLOR_BGR2GRAY);
	cv::aruco::detectMarkers(gray, dictionary, marker_corners, marker_ids);	
	
	if (marker_ids.size() > 0){
		
		cv::aruco::drawDetectedMarkers(input_image, marker_corners, marker_ids);
		for (int i = 0 ; i < marker_ids.size() ; i++){
			//~ std::cout << "Marker IDs found: " << marker_ids[i] << std::endl;
			if (marker_ids[i] == id){
				std::vector< std::vector<cv::Point2f> > single_corner(1);
				single_corner[0] = marker_corners[i];
				cv::aruco::estimatePoseSingleMarkers(single_corner, aruco_square_size, mtx, dist, rvec, tvec);
				cv::aruco::drawAxis(input_image, mtx, dist, rvec, tvec, aruco_square_size/2);
			}
		}
	}
	else{
		//~ std::cout << "No markers detected" << std::endl;
	}
}

void changeSize(int w, int h) {

	// Prevent a divide by zero, when window is too short
	// (you cant make a window of zero width).
	if (h == 0)
		h = 1;

	float ratio =  w * 1.0 / h;

	// Use the Projection Matrix
	glMatrixMode(GL_PROJECTION);

	// Reset Matrix
	glLoadIdentity();

	// Set the viewport to be the entire window
	glViewport(0, 0, w, h);

	// Set the correct perspective.
	gluPerspective(45.0f, ratio, 0.1f, 100.0f);

	// Get Back to the Modelview
	glMatrixMode(GL_MODELVIEW);
}

float angle = 0.0f;

void renderScene(void) {

	// Clear Color and Depth Buffers
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	// Reset transformations
	glLoadIdentity();
	// Set the camera
	gluLookAt(	0.0f, 0.0f, 10.0f,
				0.0f, 0.0f,  0.0f,
				0.0f, 1.0f,  0.0f);

	glRotatef(angle, 0.0f, 1.0f, 0.0f);

	glBegin(GL_TRIANGLES);
		glVertex3f(-2.0f,-2.0f, 0.0f);
		glVertex3f( 2.0f, 0.0f, 0.0);
		glVertex3f( 0.0f, 2.0f, 0.0);
	glEnd();

	angle+=0.1f;

	glutSwapBuffers();
}

int main( int argc, char** argv )
{
	// Contextual Parameters
	int target_id = 1;
	
	//~ // init GLUT and create window
	//~ glutInit(&argc, argv);
	//~ glutInitDisplayMode(GLUT_DEPTH | GLUT_DOUBLE | GLUT_RGBA);
	//~ glutInitWindowPosition(100,100);
	//~ glutInitWindowSize(320,320);
	//~ glutCreateWindow("Lighthouse3D- GLUT Tutorial");

	//~ // register callbacks
	//~ glutDisplayFunc(renderScene);
	//~ glutReshapeFunc(changeSize);
	//~ glutIdleFunc(renderScene);

	//~ // enter GLUT event processing cycle
	//~ glutMainLoop();
	
	// STEP 1: CALIBRATE CAMERA ACCORDING TO PRE-LOADED IMAGES
	camera_matrix = cv::Mat::eye(3, 3, CV_64F);
	dist_coeffs = cv::Mat::zeros(8, 1, CV_64F);
	//fixed aspect ratio
	camera_matrix.at<double>(0,0) = 1.0;
	calibrateCamera(camera_matrix, dist_coeffs, false);
	std::cout << "Calibration Matrix: " << std::endl << camera_matrix << std::endl;
	
	//STEP 2: LOAD CAMERA 
	cv::VideoCapture inputVideo;
	inputVideo.open(0);
	int framecount = 0;
	while (inputVideo.grab()) {
		cv::Mat image;
		inputVideo.retrieve(image);
		cv::Mat rvec;
		cv::Mat tvec;
		
		//STEP 3: ESTIMATE POSE
		arucoPoseEstimation(image, target_id, tvec, rvec, camera_matrix, dist_coeffs);
		std::cout << framecount << std::endl;
		framecount++;
		cv::imshow("out", image);
    char key = (char) cv::waitKey(1);
    if (key == 27)
        break;
	}
	
	return 0;
}

