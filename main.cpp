#include <stdio.h>
#include <ctype.h>
#include "opencv2/opencv.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include <iostream>
#include <time.h>
#include <stdint.h>
#include <wiringPi.h>
#include <wiringSerial.h>
#include <unistd.h>

void detect(IplImage *source, CvSize template_size, CvPoint2D32f *current_feature_p, char *status_p, int step);
void on_mouse(int event, int x, int y, int flags, void* parameters);
int bytes_per_pixel(const IplImage* image);
void process(IplImage *source, CvPoint2D32f *current_feature_p, char *status_p);
void calibration(IplImage *source);

#define WINDOW_STEP 3
#define SELECTED_REGION 2
#define SEARCHING 3
#define PREDICTING 4
#define GOOD 1
#define BAD 0
#define SELECTING_FIRST_CORNER 0
#define SELECTING_SECOND_CORNER 1

IplImage* output = 0;
//int mode = SELECTING_FIRST_CORNER;
int mode = PREDICTING;
CvPoint mouse_position;
CvPoint origin;
CvRect selection;

int main(int argc, char** argv) {
  CvCapture* capture = 0;
  IplImage*  input = 0;
  CvPoint2D32f template_point;
  CvPoint2D32f current_point;
  int p1x, p1y, p2x, p2y;
  int template_width, template_height;
  int win_step;
  char state;
  IplImage* color = 0;
  IplImage* template_color = 0;
  CvRect window;
  int tick = 0, previous_tick = 0;
  double now = 0.0;
  CvFont font;
  char buffer[256];
  uchar *pA;
  int stepA;
  uchar *data;
  CvSize sizeA;
  FILE *fp;
  clock_t start;
  clock_t passed;
  double time;
  int file_number = 0;
  char filepath[256];
  int trigger = 1;
  int fd = serialOpen("/dev/ttyACM0",115200);
  sleep(3);
  double input_array[12];
  double hidden_array_1[24];
  double hidden_array_2[24];
  int i,j;
  int count = 0;
  double predicted_array[2];
  //const unsigned char* c = reinterpret_cast<const unsigned char* >( "8500" );
  char *fname = "LinearLayer_1_W.csv";
  int ret;
  double w_1[12][24];
  double w_2[24][24];
  double w_3[24][2];
  double b_1[24];
  double b_2[24];
  double b_3[2];
  int counter_2 = 0;
  int counter_3 = 0;
  //int index_count = 0;
  int limitter = 0;
  fp = fopen( fname, "r" );
  if( fp == NULL ){
    printf("%sファイルが開けません\n", fname);
    return -1;
  }
  printf("\n");
  while((ret=fscanf(fp, "%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf", &w_1[count][0], &w_1[count][1], &w_1[count][2], &w_1[count][3], &w_1[count][4], &w_1[count][5], &w_1[count][6], &w_1[count][7], &w_1[count][8], &w_1[count][9], &w_1[count][10], &w_1[count][11], &w_1[count][12], &w_1[count][13], &w_1[count][14], &w_1[count][15], &w_1[count][16], &w_1[count][17], &w_1[count][18], &w_1[count][19], &w_1[count][20], &w_1[count][21], &w_1[count][22], &w_1[count][23])) != EOF){
    printf("%lf %lf\n", w_1[count][0], w_1[count][23]);
    count = count + 1;
  }
  count = 0;
  printf("\n");
  fclose(fp);
  fname = "LinearLayer_1_b.csv";
  fp = fopen( fname, "r" );
  if( fp == NULL ){
    printf("%sファイルが開けません\n", fname);
    return -1;
  }
  printf("\n");
  fscanf(fp, "%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf", &b_1[0], &b_1[1], &b_1[2], &b_1[3], &b_1[4], &b_1[5], &b_1[6], &b_1[7], &b_1[8], &b_1[9], &b_1[10], &b_1[11], &b_1[12], &b_1[13], &b_1[14], &b_1[15], &b_1[16], &b_1[17], &b_1[18], &b_1[19], &b_1[20], &b_1[21], &b_1[22], &b_1[23]);
  printf("%lf %lf\n", b_1[0], b_1[23]);
  printf("\n");
  fclose(fp);
  fname = "LinearLayer_2_W.csv";
  fp = fopen( fname, "r" );
  if( fp == NULL ){
    printf("%sファイルが開けません\n", fname);
    return -1;
  }
  printf("\n");
  while((ret=fscanf(fp, "%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf", &w_2[count][0], &w_2[count][1], &w_2[count][2], &w_2[count][3], &w_2[count][4], &w_2[count][5], &w_2[count][6], &w_2[count][7], &w_2[count][8], &w_2[count][9], &w_2[count][10], &w_2[count][11], &w_2[count][12], &w_2[count][13], &w_2[count][14], &w_2[count][15], &w_2[count][16], &w_2[count][17], &w_2[count][18], &w_2[count][19], &w_2[count][20], &w_2[count][21], &w_2[count][22], &w_2[count][23])) != EOF){
    printf("%lf %lf\n", w_2[count][0], w_2[count][23]);
    count = count + 1;
  }
  count = 0;
  printf("\n");
  fclose(fp);
  fname = "LinearLayer_2_b.csv";
  fp = fopen( fname, "r" );
  if( fp == NULL ){
    printf("%sファイルが開けません\n", fname);
    return -1;
  }
  printf("\n");
  fscanf(fp, "%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf", &b_2[0], &b_2[1], &b_2[2], &b_2[3], &b_2[4], &b_2[5], &b_2[6], &b_2[7], &b_2[8], &b_2[9], &b_2[10], &b_2[11], &b_2[12], &b_2[13], &b_2[14], &b_2[15], &b_2[16], &b_2[17], &b_2[18], &b_2[19], &b_2[20], &b_2[21], &b_2[22], &b_2[23]);
  printf("%lf %lf\n", b_2[0], b_2[23]);
  printf("\n");
  fclose(fp);
  fname = "LinearLayer_3_W.csv";
  fp = fopen( fname, "r" );
  if( fp == NULL ){
    printf("%sファイルが開けません\n", fname);
    return -1;
  }
  printf("\n");
  while((ret=fscanf(fp, "%lf,%lf", &w_3[count][0], &w_3[count][1])) != EOF){
    printf("%lf %lf\n", w_3[count][0], w_3[count][1]);
    count = count + 1;
  }
  count = 0;
  printf("\n");
  fclose(fp);
  fname = "LinearLayer_3_b.csv";
  fp = fopen( fname, "r" );
  if( fp == NULL ){
    printf("%sファイルが開けません\n", fname);
    return -1;
  }
  printf("\n");
  fscanf(fp, "%lf,%lf", &b_3[0], &b_3[1]);
  printf("%lf %lf\n", b_3[0], b_3[1]);
  printf("\n");
  fclose(fp);
  
  wiringPiSetup();
  fflush(stdout);
  /*if (fd<0){
    printf("can not open serialport");
  }
  serialPutchar(fd, '7');*/
    
  
  if (argc == 1 || (argc == 2 && strlen(argv[1]) == 1 && isdigit(argv[1][0]))) {
    capture = cvCreateCameraCapture(argc == 2 ? argv[1][0] - '0' : 0);
  } else if (argc == 2) {
    capture = cvCreateFileCapture(argv[1]);
  }
  if (!capture) {
    fprintf(stderr, "ERROR: capture is NULL \n");
    return (-1);
  }

  input = cvQueryFrame(capture);
  if (!input) {
    fprintf(stderr, "Could not query frame...\n");
    return (-1);
  }
  //calibration(input);
  cvGetRawData(input, &data, &stepA, &sizeA);
  printf("x_max=%d,y_max=%d", sizeA.width, sizeA.height);

  color = cvCreateImage(cvGetSize(input), IPL_DEPTH_8U, 3);
  template_color = cvCreateImage(cvGetSize(input), IPL_DEPTH_8U, 3);
  output = cvCreateImage(cvSize(input->width, input->height), IPL_DEPTH_8U, 3);
  output->origin = input->origin;
  
  cvNamedWindow("Template", 0);
  cvMoveWindow("Template", 40, 20);
  cvResizeWindow("Template", 160, 120);

  cvNamedWindow("Tracking", CV_WINDOW_AUTOSIZE);
  cvSetMouseCallback("Tracking", on_mouse, 0);
  cvMoveWindow("Tracking", 200, 100);
  cvResizeWindow("Tracking", 640, 480);

  cvInitFont(&font, CV_FONT_HERSHEY_SIMPLEX, 1.0, 1.0, 0.0, 1, 0);

  template_width = 32;
  template_height = 32;
  win_step = WINDOW_STEP;
  start = clock();
  fp = fopen("data/trajectory_0.csv","w");
  for (;;) {
    input = cvQueryFrame(capture);
    if (!input) {
      fprintf(stderr, "Could not query frame...\n");
      break;
    }

    cvCopy(input, output, NULL);
    if (mode == SEARCHING) {
      p1x = (int) current_point.x - template_width / 2 - 8 * win_step;
      p1y = (int) current_point.y - template_height / 2 - 8 * win_step;
      p2x = (int) current_point.x + template_width / 2 - 1 + 8 * win_step;
      p2y = (int) current_point.y + template_height /2 - 1 + 8 * win_step;
      cvRectangle(output, cvPoint(p1x, p1y), cvPoint(p2x, p2y), CV_RGB(0, 0, 255), 1, 8, 0);
      //detect(input, cvSize(template_width, template_height), &current_point, &state, win_step);
      process(input, &current_point, &state);
      if (((current_point.y < 100) || (current_point.y > 380 )) && trigger == 0) {
	file_number += 1;
	fclose(fp);
	sprintf(filepath, "data/trajectory_%d.csv", file_number);
	fp = fopen(filepath,"w");
	trigger = 1;
	} else if ((current_point.y < 380) && (current_point.y > 100)) {
	trigger = 0;
	//sprintf(filepath, "data/trajectory_%d.csv", file_number);
	//fp = fopen(filepath,"w");
	passed = clock();
	time = static_cast<double>(passed - start) / CLOCKS_PER_SEC * 1000;
	printf("t=%lf,x=%lf,y=%lf\n", time, current_point.x, current_point.y);
	fprintf(fp, "%f, %f, %f\n", time, current_point.x, current_point.y);
	//fclose(fp);
      }
      p1x = (int) current_point.x - template_width / 2;
      p1y = (int) current_point.y - template_height / 2;
      p2x = (int) current_point.x + template_width / 2 -1;
      p2y = (int) current_point.y + template_height / 2 -1;
      if (state == GOOD) {
	cvRectangle(output, cvPoint(p1x, p1y), cvPoint(p2x, p2y), CV_RGB(0, 255, 0), 1, 8, 0);
      } else {
	cvRectangle(output, cvPoint(p1x, p1y), cvPoint(p2x, p2y), CV_RGB(255, 0, 0), -1, 8, 0);
      }
      p1x = (int) template_point.x - template_width / 2;
      p1y = (int) template_point.y - template_height / 2;
      p2x = (int) template_point.x + template_width / 2 - 1;
      p2y = (int) template_point.y + template_height / 2 - 1;
      //cvRectangle(template_gray, cvPoint(p1x - 1, p1y - 1), cvPoint(p2x + 1, p2y + 1), CV_RGB(0, 255, 0), 1, 8, 0);
    }
    else if (mode == PREDICTING) {
      p1x = (int) current_point.x - template_width / 2 - 8 * win_step;
      p1y = (int) current_point.y - template_height / 2 - 8 * win_step;
      p2x = (int) current_point.x + template_width / 2 - 1 + 8 * win_step;
      p2y = (int) current_point.y + template_height /2 - 1 + 8 * win_step;
      cvRectangle(output, cvPoint(p1x, p1y), cvPoint(p2x, p2y), CV_RGB(0, 0, 255), 1, 8, 0);
      //detect(input, cvSize(template_width, template_height), &current_point, &state, win_step);
      process(input, &current_point, &state);
      for (i = 0; i < 12; i++){
	input_array[i] = 0;
      }
      if ((trigger == 1) && (current_point.x > 415)){
	counter_2 += 1;
	if (counter_2 == 10){
	  serialPutchar(fd, '1');
	  counter_2 = 0;
	}
      }
      if ((trigger == 1) && (current_point.x >240) && (current_point.x <290) && (current_point.y > 230) && (current_point.y < 280)){
	counter_3 += 1;
	if (counter_3 == 10){
	  serialPutchar(fd, '2');
	  counter_3 = 0;}
      }
      
	
      if ((current_point.y > 100) && trigger == 0) {
	if (count == 6){
	  while (1) {
	    predicted_array[0] = 0;
	    predicted_array[1] = 0;
	    for (i = 0; i < 24; i++) {
	      hidden_array_1[i] = 0;
	      hidden_array_2[i] = 0;
	    }
	    
	    for (i = 0; i < 24; i++) {
	      for (j = 0; j < 12; j++) {
		hidden_array_1[i] = hidden_array_1[i] + input_array[j] * w_1[j][i];
	      }
	      hidden_array_1[i] = hidden_array_1[i] + b_1[i];
	      //printf("hidden_array_1=%lf\n",hidden_array_1[i]);
	    }
	    for (i = 0; i < 24; i++) {
	      for (j = 0; j < 24; j++) {
		hidden_array_2[i] = hidden_array_2[i] + hidden_array_1[j] * w_2[j][i];
	      }
	      hidden_array_2[i] = hidden_array_2[i] + b_2[i];
	      //printf("hidden_array_2=%lf\n",hidden_array_2[i]);
	    }
	    for (i = 0; i < 2; i++) {
	      for (j = 0; j < 24; j++) {
		predicted_array[i] = predicted_array[i] + hidden_array_2[j] * w_3[j][i];
	      }
	      predicted_array[i] = predicted_array[i] + b_3[i];
	    }
	    printf("%lf,%lf\n", predicted_array[0], predicted_array[1]);
	    if (predicted_array[1] < 50) {
	      if ((predicted_array[0] < 120) && (predicted_array[0] > -60) ){
		serialPutchar(fd, '7');
		printf("sent");
		count = 0;
		trigger =1;
		break;
	      } else {
		count = 0;
		trigger = 1;
		break;
	      }
	    } else {
	      if (limitter > 40) {
		break;
	      } else {
		for (i = 0; i < 10; i++){
		  input_array[i] = input_array[i+2];}
		input_array[10] = predicted_array[0];
		input_array[11] = predicted_array[1];
		limitter += 1;
		
	      }
	    }
	    
	  }
	} else {
	  input_array[count*2+1] = 480 - current_point.y;
	  input_array[count*2] = (810 * 410) * (current_point.x - 320) / (810*410- 590*input_array[count*2+1]);
	  input_array[count*2+1] = input_array[count*2+1] * 215 / 350;
	  printf("input_array[%d]=%lf\n",count*2,input_array[count*2]);
	  printf("input_array[%d]=%lf\n",count*2+1,input_array[count*2+1]);
	  
	  count = count + 1;
	  
	}
      }
      if ((current_point.y < 100) && trigger == 1) {
	trigger = 0;
	counter_2 = 0;
	counter_3 = 0;
      }
      
      p1x = (int) current_point.x - template_width / 2;
      p1y = (int) current_point.y - template_height / 2;
      p2x = (int) current_point.x + template_width / 2 -1;
      p2y = (int) current_point.y + template_height / 2 -1;
      if (state == GOOD) {
	cvRectangle(output, cvPoint(p1x, p1y), cvPoint(p2x, p2y), CV_RGB(0, 255, 0), 1, 8, 0);
      } else {
	cvRectangle(output, cvPoint(p1x, p1y), cvPoint(p2x, p2y), CV_RGB(255, 0, 0), -1, 8, 0);
      }
      p1x = (int) template_point.x - template_width / 2;
      p1y = (int) template_point.y - template_height / 2;
      p2x = (int) template_point.x + template_width / 2 - 1;
      p2y = (int) template_point.y + template_height / 2 - 1;
      //cvRectangle(template_gray, cvPoint(p1x - 1, p1y - 1), cvPoint(p2x + 1, p2y + 1), CV_RGB(0, 255, 0), 1, 8, 0);
    }
    else if (mode == SELECTED_REGION) {
      cvGetRawData(input, &data, &stepA, &sizeA);
      pA = data;
      printf("x=%d,y=%d", selection.x, selection.y);
      pA = pA + selection.x * 3 + selection.y * stepA;
      printf("color = %d %d %d\n", pA[0], pA[1], pA[2]);
      window = selection;
      cvRectangle(output, cvPoint(window.x, window.y), cvPoint(window.x + window.width, window.y + window.height), CV_RGB(0, 255, 0), 1, 8, 0);
      template_color = input;
      template_point = cvPointTo32f(cvPoint(window.x + window.width / 2, window.y + window.height / 2));
      template_width = window.width;
      template_height = window.height;
      current_point = cvPointTo32f(cvPoint(window.x + window.width / 2, window.y + window.height / 2));
      if (current_point.y <= template_height / 2 + 8 * win_step) {
	current_point.y = template_height / 2 + 8 * win_step;
      }
      if (current_point.y >= color->height - template_height / 2 - 8 * win_step -1) {
	current_point.y = color->height - template_height / 2 - 8 * win_step -1;
      }
      if (current_point.x <= template_width / 2 + 8 * win_step) {
	current_point.x = template_width / 2 + 8 * win_step;
      }
      if (current_point.x >= color->width - template_width / 2 - 8 * win_step -1) {
	current_point.x = color->width - template_width / 2 - 8 * win_step - 1;
      }
      mode = SEARCHING;
    } else if (mode == SELECTING_SECOND_CORNER) {
      sprintf(buffer, "Select another corner!");
      cvPutText(output, buffer, cvPoint(150, 250), &font, CV_RGB(0, 255, 0));
    } else {
      sprintf(buffer, "Select a corner!");
      cvPutText(output, buffer, cvPoint(150, 250), &font, CV_RGB(0, 255, 0));
    }
    if ((mode == SELECTING_SECOND_CORNER) & (selection.width > 0) && (selection.height > 0)) {
      cvRectangle(output,cvPoint(selection.x, selection.y),cvPoint(selection.x + selection.width,selection.y + selection.height), CV_RGB(0, 0, 255), 1, 0, 0);
      cvSetImageROI(output, selection);
      cvXorS(output, cvScalarAll(255), output, 0);
      cvResetImageROI(output);
    }
    cvLine(output, cvPoint(mouse_position.x, 0), cvPoint(mouse_position.x, output->height - 1), CV_RGB(0, 255, 255), 1, 8, 0);
    cvLine(output, cvPoint(0, mouse_position.y), cvPoint(output->width - 1, mouse_position.y), CV_RGB(0, 255, 255), 1, 8, 0);
    sprintf(buffer, "%3.1lfms", now / 1000);
    cvPutText(output, buffer, cvPoint(50, 150), &font, CV_RGB(255, 0, 0));
    cvShowImage("Tracking", output);
    if (template_color) {
      cvShowImage("Template", template_color);
    }
    if (cvWaitKey(10) >= 0) {
      break;
    }

    tick = cvGetTickCount();
    now = (tick - previous_tick) / cvGetTickFrequency();
    previous_tick = tick;
  }

  fclose(fp);
  cvReleaseImage(&template_color);
  cvReleaseImage(&color);
  cvReleaseImage(&output);
  cvReleaseCapture(&capture);
  cvDestroyWindow("Tracking");
  return 0;
    
 
    
}
int bytes_per_pixel(const IplImage* image) {
  return ((((image)->depth & 255) / 8) * (image)->nChannels);
}
void process(IplImage *source, CvPoint2D32f *current_feature_p, char *status_p) {
  int bppS;
  uchar *pS;
  uchar *dataS;
  int stepS;
  CvSize sizeS;
  int x, y;
  int count = 0;

  bppS = bytes_per_pixel(source);
  

  cvGetRawData(source, &dataS, &stepS, &sizeS);

  pS = dataS;
  *status_p = BAD;

  for (y = 0; y < sizeS.height; y++) {
    for (x = 0; x < sizeS.width; x++) {
      if ((pS[0] < 70) && (pS[1] < 50) && (pS[2] > 100)) {
        count += 1;
      } else {
	count = 0;
      }
      if (count > 0){
	(*current_feature_p).x = x;
	(*current_feature_p).y = y;
	*status_p = GOOD;
	break;
      }
      pS += bppS;
    }
    if (count > 0){
      break;
    }
  }
}

void calibration(IplImage *source) {
  int bppS;
  uchar *pS;
  uchar *dataS;
  int stepS;
  CvSize sizeS;
  int x, y;
  FILE *fp;
  fp = fopen("calib_dataset.csv", "w");
  bppS = bytes_per_pixel(source);
  cvGetRawData(source, &dataS, &stepS, &sizeS);
  pS = dataS;
  for (y = 0; y < sizeS.height; y++) {
    for (x = 0; x < sizeS.width; x++) {
      if ((pS[0] > 75) && (pS[1] < 50) && (pS[2] < 40)) {
	fprintf(fp, "%d, %d\n", x, y);
      }
      pS += bppS;
    }
  }
  fclose(fp);
}
void detect(IplImage *source, CvSize template_size, CvPoint2D32f *current_feature_p, char *status_p, int step) {
  int bppS;
  uchar *pS;
  uchar *dataS;
  int stepS;
  CvSize sizeS;
  int count = 0;
  float sum = 0;
  int u, v, uu, vv;

  bppS = bytes_per_pixel(source);
  cvGetRawData(source, &dataS, &stepS, &sizeS);
  *status_p = BAD;

  for (v = (int) (*current_feature_p).y - 8 * step;
       v < (int) (*current_feature_p).y + 8 * step; v += step) {
    for (u = (int) (*current_feature_p).x - 8 * step;
	 u < (int) (*current_feature_p).x + 8 * step; u += step) {
      sum = 0;
      count = 0;
      for (vv = -template_size.height / 2; vv < template_size.height / 2; vv += step) {
	for (uu = -template_size.width / 2; uu < template_size.width / 2; uu += step) {
	  pS = dataS + bppS * (u + uu) + stepS * (v + vv);
	  if ((pS[0] < 60) && (pS[1] < 50) && (pS[2] > 75)){
	    sum += 1;}
	  count += 1;
	  
	}
      }
      if ((sum / count) >  0.7){
	(*current_feature_p).x = u;
	(*current_feature_p).y = v;
	*status_p = GOOD;
	break;
      }
    }
    if ((sum / count) > 0.7){
      break;
    }
  }
  
  
}

void on_mouse(int event, int x, int y, int flags, void* parameters) {
  if (!output) {
    return;
  }
  if (output->origin) {
    y = (output->height) - y;
  }
  if (mode == SELECTING_SECOND_CORNER) {
    selection.x = MIN(x, origin.x);
    selection.y = MIN(y, origin.y);
    selection.width = selection.x + CV_IABS(x - origin.x);
    selection.height = selection.y + CV_IABS(y - origin.y);
    selection.x = MAX(selection.x, 0);
    selection.y = MAX(selection.y, 0);
    selection.width = MIN(selection.width, output->width);
    selection.height = MIN(selection.height, output->height);
    selection.width = selection.width - selection.x;
    selection.height = selection.height - selection.y;
  }

  switch (event) {
  case CV_EVENT_MOUSEMOVE:
    mouse_position = cvPoint(x, y);
    break;
  case CV_EVENT_LBUTTONDOWN:
    if (mode == SEARCHING) {
      mode = SELECTING_FIRST_CORNER;
    }
    origin = cvPoint(x, y);
    selection = cvRect(x, y, 0, 0);
    mode = SELECTING_SECOND_CORNER;
    break;
  case CV_EVENT_LBUTTONUP:
    if ((selection.width > 0) && (selection.height > 0)) {
      mode = SELECTED_REGION;
    } else {
      mode = SELECTING_FIRST_CORNER;
    }
    break;
  }
}
