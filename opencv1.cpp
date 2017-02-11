#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/aruco.hpp>
#include <iostream>
#include <sstream>
#include <time.h>
#include <stdio.h>
#include <thread>
#include <math.h>
#include <cmath>
#include "ConfigFile.cpp"

#ifdef __APPLE__
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#endif

// Create checkerboard texture 
#define checkImageWidth 64
#define checkImageHeight 64
static GLubyte checkImage[checkImageHeight][checkImageWidth][4];
static GLuint texName;
cv::Mat mat_vertex;

void makeCheckImage(void){
	int i, j, c;
	for (i = 0; i < checkImageHeight; i++) {
		for (j = 0; j < checkImageWidth; j++) {
		c = ((((i&0x8)==0)^((j&0x8))==0))*255;
		checkImage[i][j][0] = (GLubyte) c;
		checkImage[i][j][1] = (GLubyte) c;
		checkImage[i][j][2] = (GLubyte) c;
		checkImage[i][j][3] = (GLubyte) 255;
		}
	}
}

// Contextual Parameters
int target_id = 1;
cv::Ptr<cv::aruco::Dictionary> dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);
cv::Mat camera_matrix;
cv::Mat dist_coeffs;
float angle = 0.0f;
float triangle_size = 0.082;
int framecount = 0;
cv::Mat graphic_tvec;
cv::Mat graphic_rvec;
bool transformation_ready = false;
std::string calibration_file = "calibration3";
std::vector<std::vector<float> > vertex;
std::vector<std::vector<float> > vertex_color;

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
	cv::Mat first_board_image;
	first_board_image = cv::imread(calibration_file + "/0.jpg", CV_LOAD_IMAGE_COLOR);   
	calibration_image_size.width = first_board_image.cols;
	calibration_image_size.height = first_board_image.rows;
	std::cout << "Calibration image size: " << first_board_image.cols << " " << first_board_image.rows << std::endl;
	
	// Contextual Parameters
	board_size.width = 9;
	board_size.height = 6;
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
		board_image = cv::imread(calibration_file + "/" + s + ".jpg", CV_LOAD_IMAGE_COLOR);   
		
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
		std::cout << "Reprojection-Error from calibration: " << rms << std::endl;
	}
	
}

bool arucoPoseEstimation(cv::Mat& input_image, int id, cv::Mat& tvec, cv::Mat& rvec, cv::Mat& mtx, cv::Mat& dist, bool draw_axis){
	// Contextual Parameters
	//~ std::cout << "Pose estimation called" << std::endl;
	float aruco_square_size = 0.082;
	bool marker_found = false;
	std::vector< int > marker_ids;
	std::vector< std::vector<cv::Point2f> > marker_corners, rejected_candidates;
	cv::Mat gray;
	
	cv::cvtColor(input_image, gray, cv::COLOR_BGR2GRAY);
	cv::aruco::detectMarkers(gray, dictionary, marker_corners, marker_ids);	
	
	if (marker_ids.size() > 0){
		for (int i = 0 ; i < marker_ids.size() ; i++){
			//~ std::cout << "Marker IDs found: " << marker_ids[i] << std::endl;
			if (marker_ids[i] == id){
				std::vector< std::vector<cv::Point2f> > single_corner(1);
				single_corner[0] = marker_corners[i];
				cv::aruco::estimatePoseSingleMarkers(single_corner, aruco_square_size, mtx, dist, rvec, tvec);
				if (draw_axis){
					cv::aruco::drawDetectedMarkers(input_image, marker_corners, marker_ids);
					cv::aruco::drawAxis(input_image, mtx, dist, rvec, tvec, aruco_square_size/2);
				}
				//~ std::cout << "Marker found : aruco pose estimation" << std::endl;
				marker_found = true;
			}
		}
	}
	else{
		//~ std::cout << "No markers detected" << std::endl;
	}
	
	return marker_found;
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

void init_texture(void){
	
	std::cout << "Initiating Vertex" << std::endl;
	
	//~ glClearColor (0.0, 0.0, 0.0, 0.0);
	//~ glShadeModel(GL_FLAT);
	//~ glEnable(GL_DEPTH_TEST);
	//~ makeCheckImage();
	//~ glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
	//~ glGenTextures(1, &texName);
	//~ glBindTexture(GL_TEXTURE_2D, texName);
	//~ glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_S, GL_REPEAT);
	//~ glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_T, GL_REPEAT);
	//~ glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_NEAREST);
	//~ glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_NEAREST);
	//~ glTexImage2D(GL_TEXTURE_2D, 0,GL_RGBA, checkImageWidth,checkImageHeight,0, GL_RGBA, GL_UNSIGNED_BYTE,checkImage);
	
	cv::flip(mat_vertex, mat_vertex, 0);
	glGenTextures(1, &texName);
	glBindTexture(GL_TEXTURE_2D, texName);
	
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
	glTexImage2D(GL_TEXTURE_2D,     // Type of texture
								 0,                 // Pyramid level (for mip-mapping) - 0 is the top level
								 GL_RGB,            // Internal colour format to convert to
								 mat_vertex.cols,          // Image width  i.e. 640 for Kinect in standard mode
								 mat_vertex.rows,          // Image height i.e. 480 for Kinect in standard mode
								 0,                 // Border width in pixels (can either be 1 or 0)
								 GL_BGR, // Input image format (i.e. GL_RGB, GL_RGBA, GL_BGR etc.)
								 GL_UNSIGNED_BYTE,  // Image data type
								 mat_vertex.ptr());        // The actual image data itself

}

void update_texture(void){
	glTexImage2D(GL_TEXTURE_2D,     // Type of texture
								 0,                 // Pyramid level (for mip-mapping) - 0 is the top level
								 GL_RGB,            // Internal colour format to convert to
								 mat_vertex.cols,          // Image width  i.e. 640 for Kinect in standard mode
								 mat_vertex.rows,          // Image height i.e. 480 for Kinect in standard mode
								 0,                 // Border width in pixels (can either be 1 or 0)
								 GL_BGR, // Input image format (i.e. GL_RGB, GL_RGBA, GL_BGR etc.)
								 GL_UNSIGNED_BYTE,  // Image data type
								 mat_vertex.ptr());        // The actual image data itself
}

void renderScene(void) {
	
	// Texturing
	update_texture();
	
	// Clear Color and Depth Buffers
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	// Reset transformations
	glLoadIdentity();
	// Set the camera
	gluLookAt(	0.0f, 0.0f, 10.0f,
				0.0f, 0.0f,  0.0f,
				0.0f, 1.0f,  0.0f);
		
	// Texturing
	glEnable(GL_TEXTURE_2D);
	glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_DECAL);
	glBindTexture(GL_TEXTURE_2D, texName);
	
	glBegin(GL_QUADS);
		
	 glTexCoord2f(0.0, 1.0);
   glVertex3f(-5.0f, -5.0f, 0.0f);
   glTexCoord2f(0.0, 0.0);
   glVertex3f(-5.0f, 5.0f, 0.0f);
   glTexCoord2f(1.0, 0.0);
   glVertex3f(5.0f, 5.0f, 0.0f);
   glTexCoord2f(1.0, 1.0);
   glVertex3f(5.0f, -5.0f, 0.0f);
  
  glEnd();
  
  // Texturing
	glDisable(GL_TEXTURE_2D);
	
	//~ glRotatef(angle, 0.0f, 1.0f, 0.0f);
	cv::Mat tvec;
	cv::Mat rvec;
	if (transformation_ready==true){		
		tvec = graphic_tvec;
		glTranslatef(tvec.at<double>(0), -tvec.at<double>(1), -tvec.at<double>(2));
		
		rvec = graphic_rvec;
		float a = rvec.at<double>(0);
		float b = rvec.at<double>(1);
		float c = rvec.at<double>(2);
		float theta = sqrt(pow(a,2) + pow(b,2) + pow(c,2));
		glRotatef(-theta*180/M_PI, -a/theta, b/theta, c/theta);
		//~ std::cout << "Rotation angle: " << theta << std::endl;
	}
	int scount = 0;
	
	glBegin(GL_QUADS); // of the color cube
 
   //~ // Top-face
   glColor3f(0.0f, 1.0f, 0.0f); // green
   glVertex3f(1.0f, 1.0f, -1.0f);
   glVertex3f(-1.0f, 1.0f, -1.0f);
   glVertex3f(-1.0f, 1.0f, 1.0f);
   glVertex3f(1.0f, 1.0f, 1.0f);
   
   //~ // Bottom-face
   glColor3f(1.0f, 0.5f, 0.0f); // orange
   glVertex3f(1.0f, -1.0f, 1.0f);
   glVertex3f(-1.0f, -1.0f, 1.0f);
   glVertex3f(-1.0f, -1.0f, -1.0f);
   glVertex3f(1.0f, -1.0f, -1.0f);
 
   // Front-face
   glColor3f(1.0f, 0.0f, 0.0f); // red
   glVertex3f(1.0f, 1.0f, 1.0f);
   glVertex3f(-1.0f, 1.0f, 1.0f);
   glVertex3f(-1.0f, -1.0f, 1.0f);
   glVertex3f(1.0f, -1.0f, 1.0f);
 
   // Back-face
   glColor3f(1.0f, 1.0f, 0.0f); // yellow
   glVertex3f(1.0f, -1.0f, -1.0f);
   glVertex3f(-1.0f, -1.0f, -1.0f);
   glVertex3f(-1.0f, 1.0f, -1.0f);
   glVertex3f(1.0f, 1.0f, -1.0f);
 
   // Left-face
   glColor3f(0.0f, 0.0f, 1.0f); // blue
   glVertex3f(-1.0f, 1.0f, 1.0f);
   glVertex3f(-1.0f, 1.0f, -1.0f);
   glVertex3f(-1.0f, -1.0f, -1.0f);
   glVertex3f(-1.0f, -1.0f, 1.0f);
 
   // Right-face
   glColor3f(1.0f, 0.0f, 1.0f); // magenta
   glVertex3f(1.0f, 1.0f, -1.0f);
   glVertex3f(1.0f, 1.0f, 1.0f);
   glVertex3f(1.0f, -1.0f, 1.0f);
   glVertex3f(1.0f, -1.0f, -1.0f);
 
	glEnd();

	//~ glBegin(GL_TRIANGLES);
		
		//~ glColor3f(1.0f, 0.0f, 0.0f);     // Red	
		//~ glVertex3f(0.0, 0.0, 0.0);	
		//~ glVertex3f(1.0, 0.0, 0.0);	
		//~ glVertex3f(0.0, 0.0, 1.0);	
		
		//~ glColor3f(0.0f, 1.0f, 0.0f);     // Green
		//~ glVertex3f(0.0, 0.0, 1.0);	
		//~ glVertex3f(1.0, 0.0, 0.0);	
		//~ glVertex3f(0.0, 1.0, 1.0);	
		
		//~ glColor3f(0.0f, 0.0f, 1.0f);     // Blue
		//~ glVertex3f(0.0, 1.0, 1.0);	
		//~ glVertex3f(0.0, 0.0, 1.0);	
		//~ glVertex3f(0.0, 0.0, 0.0);	
		
		//~ glColor3f(0.0f, 0.0f, 1.0f);     // Yellow
		//~ glVertex3f(0.0, 1.0, 1.0);	
		//~ glVertex3f(0.0, 0.0, 1.0);	
		//~ glVertex3f(1.0, 0.0, 0.0);	
		
		//~ // Front
		//~ glColor3f(1.0f, 0.0f, 0.0f);     // Red
		//~ glTexCoord2f(0.0, 0.0);
		//~ glVertex3f( 0.0f, 1.0f, 0.0f);
		//~ glColor3f(0.0f, 1.0f, 0.0f);     // Green
		//~ glTexCoord2f(1.0, 0.0);
		//~ glVertex3f(-1.0f, -1.0f, 1.0f);
		//~ glColor3f(0.0f, 0.0f, 1.0f);     // Blue
		//~ glTexCoord2f(1.0, 1.0);
		//~ glVertex3f(1.0f, -1.0f, 1.0f);

		//~ // Right
		//~ glColor3f(1.0f, 0.0f, 0.0f);     // Red
		//~ glVertex3f(0.0f, 1.0f, 0.0f);
		//~ glColor3f(0.0f, 0.0f, 1.0f);     // Blue
		//~ glVertex3f(1.0f, -1.0f, 1.0f);
		//~ glColor3f(0.0f, 1.0f, 0.0f);     // Green
		//~ glVertex3f(1.0f, -1.0f, -1.0f);

		//~ // Back
		//~ glColor3f(1.0f, 0.0f, 0.0f);     // Red
		//~ glVertex3f(0.0f, 1.0f, 0.0f);
		//~ glColor3f(0.0f, 1.0f, 0.0f);     // Green
		//~ glVertex3f(1.0f, -1.0f, -1.0f);
		//~ glColor3f(0.0f, 0.0f, 1.0f);     // Blue
		//~ glVertex3f(-1.0f, -1.0f, -1.0f);

		//~ // Left
		//~ glColor3f(1.0f,0.0f,0.0f);       // Red
		//~ glVertex3f( 0.0f, 1.0f, 0.0f);
		//~ glColor3f(0.0f,0.0f,1.0f);       // Blue
		//~ glVertex3f(-1.0f,-1.0f,-1.0f);
		//~ glColor3f(0.0f,1.0f,0.0f);       // Green
		//~ glVertex3f(-1.0f,-1.0f, 1.0f);
		
		// CODE FOR RENDERING MODEL FROM .TXT FILE
		//~ glColor3f(vertex_color[scount][0],vertex_color[scount][1],vertex_color[scount][2]);
		//~ scount++;
		//~ for (int i = 0 ; i < vertex.size() ; i++){
			
			//~ if (((i+1) % 3) == 0){
				//~ glColor3f(vertex_color[scount][0],vertex_color[scount][1],vertex_color[scount][2]);
				//~ scount++;
			//~ }
			
			//~ glVertex3f(vertex[i][0],vertex[i][1], vertex[i][2]);
			//~ std::cout << i << std::endl;
		//~ }
		
	//~ glEnd();

	glutSwapBuffers();
	
}

// Function copied from stackOverflow
std::string type2str(int type) {
  std::string r;

  uchar depth = type & CV_MAT_DEPTH_MASK;
  uchar chans = 1 + (type >> CV_CN_SHIFT);

  switch ( depth ) {
    case CV_8U:  r = "8U"; break;
    case CV_8S:  r = "8S"; break;
    case CV_16U: r = "16U"; break;
    case CV_16S: r = "16S"; break;
    case CV_32S: r = "32S"; break;
    case CV_32F: r = "32F"; break;
    case CV_64F: r = "64F"; break;
    default:     r = "User"; break;
  }

  r += "C";
  r += (chans+'0');

  return r;
}

// Function not completed
void mat_to_glubyte(cv::Mat& mat){
	//~ cv::imshow("out", mat);
	//~ cv::waitKey(0);
		
	std::cout << "Performing conversion" << std::endl;
	std::string mat_type = type2str(mat.type());
	std::cout << "Matrix type: " << mat_type << std::endl;
	std::cout << "Matrix size: " << mat.size() << std::endl;
	
	for (int i = 0 ; i < mat.size().width ; i++){
		for (int j = 0; j < mat.size().height ; j++){
			std::cout << "i: " << i << " , j: " << j << std::endl;
			std::cout << "test: " << mat.at<char>(0,0) << std::endl;
			//~ mat_texutre[i][j][0] = (GLubyte) (mat.at<double>(i,j,0));
			//~ mat_texutre[i][j][1] = (GLubyte) (mat.at<double>(i,j,1));
			//~ mat_texutre[i][j][2] = (GLubyte) (mat.at<double>(i,j,2));
			//~ mat_texutre[i][j][3] = (GLubyte) 255;
		}
	}
}

void load_static_texture(void) {
	std::cout << std::endl << std::endl << "Creating static mat texture" << std::endl;
	mat_vertex = cv::imread("test/0.png");  
	std::cout << "Image size: " << mat_vertex.size() << std::endl;
	//~ mat_to_glubyte(image_retrieve);
}

// THREAD FUNCTIONS
void *start_gl_main(void *threadid){
	long tid;
  tid = (long)threadid;
  std::cout << "Start gl main thread id : " << tid << std::endl;
	glutMainLoop();
	pthread_exit(NULL);
}

void *start_cv_main(void *threadid){
	cv::VideoCapture inputVideo;
	inputVideo.open(0);
	bool marker_found;
	while (inputVideo.grab()) {
		cv::Mat image;
		cv::Mat rvec;
		cv::Mat tvec;
		inputVideo.retrieve(image);
		
		mat_vertex = image;
		
		marker_found = false;
		marker_found = arucoPoseEstimation(image, target_id, tvec, rvec, camera_matrix, dist_coeffs, true);
		if (marker_found==true){
			//~ std::cout << "Marker found" << std::endl;
			//~ std::cout << tvec << std::endl;
			//~ std::cout << tvec.at<double>(0) << std::endl;
			transformation_ready = true;
			// Set global translation vector
			graphic_tvec = tvec;
			
			// Set global rotation vector
			graphic_rvec = rvec;
		}
		//~ cv::imshow("out", image);
		//~ char key = (char) cv::waitKey(1);
	  //~ if (key == 27){
			//~ break;
			//~ inputVideo.release();
			//~ pthread_exit(NULL);
		//~ }
	}
}

int main( int argc, char** argv )
{
	// CALIBRATE CAMERA
	camera_matrix = cv::Mat::eye(3, 3, CV_64F);
	dist_coeffs = cv::Mat::zeros(8, 1, CV_64F);
	//fixed aspect ratio
	camera_matrix.at<double>(0,0) = 1.0;
	calibrateCamera(camera_matrix, dist_coeffs, false);
	std::cout << "Calibration Matrix: " << std::endl << camera_matrix << std::endl;
	
	// LOAD GL MODEL
	// CODE FOR RENDERING MODEL FROM .TXT FILE
	//~ ConfigFile cf("model/model2.txt");
	//~ cf.getVertex(vertex);
	//~ std::cout << "Size of vertex: " << vertex.size() << std::endl;
	
	//~ ConfigFile cf2("model/color.txt");
	//~ cf2.getVertex(vertex_color);
	
	// MAT texturing test
	load_static_texture();
	
	// init GLUT and create window
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DEPTH | GLUT_DOUBLE | GLUT_RGBA);
	glutInitWindowPosition(100,100);
	glutInitWindowSize(320,320);
	glutCreateWindow("Lighthouse3D- GLUT Tutorial");
	init_texture();
	
	// register callbacks
	glutDisplayFunc(renderScene);
	glutReshapeFunc(changeSize);
	glutIdleFunc(renderScene);

	pthread_t thread[2];
	int threadError;
	int i=0;

	// START GL THREAD
	threadError = pthread_create(&thread[i], NULL, start_gl_main, (void *)i);

	if (threadError){
		std::cout << "Error:unable to create thread," << threadError << std::endl;
		exit(-1);
	}

	// START CV THREAD
	i++;
	threadError = pthread_create(&thread[i], NULL, start_cv_main, (void *)i);

	if (threadError){
		std::cout << "Error:unable to create thread," << threadError << std::endl;
		exit(-1);
	}
	
	pthread_exit(NULL);
	
	return 0;
}

