
#include <iostream>


#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/line_descriptor/descriptor.hpp>

//#include <opencv2/contrib/contrib.hpp>

using namespace cv;
using namespace std;

//=======================================================================================
// computeHistogram
//=======================================================================================
void computeHistogram(const Mat& inputComponent, Mat& myHist)
{
	/// Establish the number of bins
	int histSize = 256;
	/// Set the ranges ( for B,G,R) )
	float range[] = { 0, 256 } ;
	const float* histRange = { range };
	bool uniform = true; 
	bool accumulate = false;
	
	/// Compute the histograms:
	calcHist( &inputComponent, 1, 0, Mat(), myHist, 1, &histSize, &histRange, uniform, accumulate );
}

//=======================================================================================
// displayHistogram
//=======================================================================================
void displayHistogram(const Mat& myHist)
{
	// Establish the number of bins
	int histSize = 256;	
	// Draw one histogram
	int hist_w = 512; int hist_h = 400;
	int bin_w = cvRound( (double) hist_w/histSize );
	Mat histImage( hist_h, hist_w, CV_8UC3, Scalar( 0,0,0) );
	/// Normalize the result to [ 0, histImage.rows ]
	Mat myHistNorm;
	normalize(myHist, myHistNorm, 0, histImage.rows, NORM_MINMAX, -1, Mat() );

	/// Draw for each channel
	for( int i = 1; i < histSize; i++ )
	{
		line( histImage, Point( bin_w*(i-1), hist_h - cvRound(myHistNorm.at<float>(i-1)) ) , Point( bin_w*(i), hist_h - cvRound(myHistNorm.at<float>(i)) ), Scalar( 255, 255, 255), 2, 8, 0 );		
	}
	/// Display
	namedWindow("Display Histo", CV_WINDOW_AUTOSIZE );
	imshow("Display Histo", histImage );
	//imwrite("hist.jpg", histImage);
	cvWaitKey();
}

//=======================================================================================
// Mat norm_0_255(InputArray _src)
// Create and return normalized image
//=======================================================================================
Mat norm_0_255(InputArray _src) {
 Mat src = _src.getMat();
 // Create and return normalized image:
 Mat dst;
 switch(src.channels()) {
	case 1:
		normalize(_src, dst, 0, 255, NORM_MINMAX, CV_8UC1);
		break;
	case 3:
		normalize(_src, dst, 0, 255, NORM_MINMAX, CV_8UC3);
		break;
	default:
	src.copyTo(dst);
	break;
 }
 return dst;
}

//=======================================================================================
// EQM
//=======================================================================================
double eqm(const Mat & img1, const Mat & img2)
{
	// Récupération de la longueur et largeur de l'image
	int width = img1.cols;
	int height = img1.rows;

	// Test si nous avons bien une image en paramètre
	if(width == 0 || height == 0)
	{
		cout << "Erreur taille";
		return -1;
	}
	
	double eqm = 0;
	// Calcul de l'erreur quadratique moyenne en faisant une somme des différences sur chaque pixel entre les deux images au carré
	for(int i=0;i<height;i++)
	{
		for(int j=0;j<width;j++)
		{
			eqm += pow(img1.at<uchar>(i,j) - img2.at<uchar>(i,j),2); 
		}
	}
	// Puis on normalise en divisant par le nombre de pixel dans l'image
	eqm *= (double)1/(width*height);
	
 	return eqm;
}

//=======================================================================================
// psnr
//=======================================================================================
double psnr(const Mat & imgSrc, const Mat & imgDeg)
{
	// Calcul de D qui est la dynamique couverte par les variations de l'intensité lumineuse soit 256 -1 = 255 valeurs possibles
	double D = pow(2,8)-1;
	// Récupération de l'eqm entre l'image originale et dégradée
	double eqmVal = eqm(imgSrc,imgDeg);
	cout << "Eqm : " << eqmVal << " \n";;
	
	// Test si l'eqm est égale à 0 (passage de 2 images identiques) -> impossible de calculer le psnr
	if (eqmVal==0)
	{
		cout<< "Eqm nulle " << "\n";
		return NULL;
	}
	// On retourne la valeur du psnr
 	return 10*log10(pow(D,2)/eqmVal);
}

//=======================================================================================
// distortionMap
//=======================================================================================
void distortionMap(const vector<Mat> & imgSrc, const vector<Mat> & imgDeg, Mat &distoMap)
{
	// Test sur la taille des 2 images (elles doivent avoir la même taille)
	assert(imgSrc[0].size() == imgDeg[0].size());
	// Calcul de la carte d'erreur sur le canal Y des 2 images (différence entre des 2 canaux Y des images pixels à pixels)
	// (Contient des valeurs négatives)
	// On rajoute 128 afin de recentrer les valeurs sur 128 au lieu de 0 (on suppose que les valeurs ne sont pas différentes a plus 128 valeurs de gris)
	distoMap= imgSrc[0] - imgDeg[0] +128;
}

//=======================================================================================
// entropy
//=======================================================================================
void entropy(const Mat& myHist, int tailleimage, double &result)
{
	result = 0;

	// Pour chaque valeur de l'histogramme (histogramme organisé sous forme de matrice colonne de float)
	// On calcul l'entropy qui va etre égale à moins la somme des probabilités d'obtenir la valeur en i (i allant de 0 à 255) x le log base 2 de la probabilité
	for (int i = 0; i < myHist.rows; i++)
	{
		// Si la valeur n'est pas égale à 0 alors on l'ajoute à notre calcul de l'entropy
		// Si la valeur est égale à 0 la multiplication est normalement égale à 0 c'est pourquoi cette valeur peut être ignorée
		if (myHist.at<float>(i, 0) != 0)
		{
			// Afin de calculer la probabilité d'appartion de la valeur en i il suffit de divisé le nombre d'apparition de cette valeur par le nombre de pixels dans l'image
			result = result - myHist.at<float>(i,0)/tailleimage * log2(myHist.at<float>(i,0) /tailleimage);
		}
	}
}

//=======================================================================================
// énergie
//=======================================================================================
void energy(const Mat& im, double &result)
{
	// Récupération de la longueur et largeur de l'image
	int width = im.cols;
	int height = im.rows;

	result = 0;
	for (int i = 0; i<height; i++)
	{
		for (int j = 0; j<width; j++)
		{
			result += pow(im.at<uchar>(i, j), 2);
		}
	}
}

//=======================================================================================
// calculErreurBlock
//=======================================================================================
void calculErreurBlock(const Mat & It, const Mat & It1, int i, int j, int sizeBlock, int nbBlock, int numCrit, int & erreurMin, Mat & imgErreur, Mat & imgVecteur){

	int h,w;
	h = It.rows;
	w = It.cols;
	
	int N;

	Vec2i distance;
	
	int nbPix = nbBlock * sizeBlock;
	N=nbPix/2;
	int coordWini, coordWinj;
	int bestCoordi, bestCoordj;
	//Initialisation de l'erreurMin
	erreurMin=std::numeric_limits<int>::max();
	int erreurpix;
	int erreur = 0;
	
	//Parcours de la fenêtre dans l'image It
	for(int s=0; s<nbPix; s++){
		for(int t=0; t<nbPix; t++){
			//initialisation de l'erreur à chaque itération
			erreur = 0;
			//Coordonées dans la fenêtre
			coordWini = i-N+s;
			coordWinj = j-N+t; 
			if(coordWini>0 && coordWinj>0 &&coordWini<h && coordWinj<w){
				for (int ib=0; ib<sizeBlock; ib++){
					for(int jb=0; jb<sizeBlock; jb++){
						//Calcul de l'erreur sur les blocks considérés
						switch (numCrit){ 
							case 0 : 
								erreur += abs((int)It.at<uchar>(coordWini+ib,coordWinj+jb)-(int)It1.at<uchar>(i+ib,j+jb));
								break;
							case 1 : 
								erreur += ((int)It.at<uchar>(coordWini+ib,coordWinj+jb)-(int)It1.at<uchar>(i+ib,j+jb))*((int)It.at<uchar>(coordWini + ib, coordWinj + jb) - (int)It1.at<uchar>(i + ib, j + jb));
								break;
							default : 
								erreurpix = abs((int)It.at<uchar>(coordWini+ib,coordWinj+jb)-(int)It1.at<uchar>(i+ib,j+jb));
								if(erreur < erreurpix) erreur = erreurpix;   
						}
					}
	
				}
				
				//Sauvegarde de la position du meilleur block
				if(erreur<erreurMin){
					 erreurMin = erreur;
					 distance[0] = i-coordWini;
					 bestCoordi = coordWini;
					 distance[1] = j-coordWinj;
					 bestCoordj = coordWinj;
					 }
				
			}
			
		}
	}
	
	//Création de la carte d'erreur et de mouvement
	for (int ib=0; ib<sizeBlock; ib++){
			for(int jb=0; jb<sizeBlock; jb++){
					imgErreur.at<int>(i+ib,j+jb) = (int)It.at<uchar>(bestCoordi+ib,bestCoordj+jb)-(int)It1.at<uchar>(i+ib,j+jb);
					imgVecteur.at<Vec2i>(i+ib,j+jb) = distance;
			}
	}
	
}

//=======================================================================================
// calculMouvement
//=======================================================================================
void calculMouvement(const Mat & It, const Mat & It1, int sizeBlock, int nbBlock, int numCrit, Mat & imgErreur, Mat & imgVecteur){
	
	int h,w;
	h = It.rows;
	w= It.cols;
	
	int erreur;
	for (int i=0; i<h; i=i+sizeBlock){
		for(int j=0; j<w; j=j+sizeBlock){
			calculErreurBlock(It, It1, i, j, sizeBlock, nbBlock, numCrit, erreur, imgErreur, imgVecteur); //Appel la fonction calculErreurBlock pour chaque pixel de l'image		
		}
	}
	
}

//=======================================================================================
// reconstruction
//=======================================================================================
void reconstruction(Mat & It1, const Mat & It, const Mat & imgErreur, const Mat & imgVecteur, int sizeBlock) {
	int h, w;
	h = It.rows;
	w = It.cols;

	int N;

	Vec2i distance;
	
	//parcours de l'image et décodage en allant chercher la valeur initiale grâce à la distance puis en soustrayant l'erreur
	for (int i = 0; i < h; i=i+sizeBlock) {
		for (int j = 0; j < w; j=j+sizeBlock) {
			distance[0] = imgVecteur.at<Vec2i>(i,j)[0];
			distance[1] = imgVecteur.at<Vec2i>(i,j)[1];
			for (int s = 0; s < sizeBlock; s++) {
				for (int t = 0; t < sizeBlock; t++) {
					It1.at<int>(i + s, j + t) = (int) It.at<uchar>(i-distance[0] + s,j-distance[1] + t) - imgErreur.at<int>(i + s, j + t);					
				}
			}
		
		}
		
	}
}

//=======================================================================================
// normMax
//=======================================================================================
double normMax(const Mat & imgVecteur) {
	double normMaxi = std::numeric_limits<int>::min();
	double norm;
	for (int i = 0; i < imgVecteur.rows; i++) {
		for (int j = 0; j < imgVecteur.cols; j++) {
			norm= sqrt(pow(imgVecteur.at<Vec2i>(i, j)[0],2)+ pow(imgVecteur.at<Vec2i>(i, j)[1],2));
			if (norm > normMaxi) normMaxi = norm;
		}
	}
	return normMaxi;
}
//=======================================================================================
//=======================================================================================
// MAIN
//=======================================================================================
//=======================================================================================
int main(int argc, char** argv){

	// Lecture des 2 fichiers sans passer par les arguments fournis
	Mat inputImageSrc = imread("../../image7.jpg", CV_LOAD_IMAGE_COLOR);
	Mat inputImageSrc2 = imread("../../image9.jpg", CV_LOAD_IMAGE_COLOR);

	// Images qui détiendront les images en YCrCb
	Mat convertImageSrc;
	Mat convertImageSrc2;
  	
	// Conversion des 2 images RGB en image YCrCb
	cvtColor(inputImageSrc, convertImageSrc, COLOR_BGR2YCrCb);
	cvtColor(inputImageSrc2, convertImageSrc2, COLOR_BGR2YCrCb);
  
	// Vecteurs d'image qui détiendront les valeurs Y Cr et Cb des 2 images
	vector<Mat> vectorInput;
	vector<Mat> vectorInput2;
  
	// Découpage des 3 composantes Y Cr et Cb dans des vecteurs d'image des 2 images
	split(inputImageSrc, vectorInput);
	split(inputImageSrc2, vectorInput2);

	//=======================================================================================
	// Main
	//=======================================================================================
	
	int h,w;
	h = vectorInput[0].rows;
	w = vectorInput[0].cols;
	int sizeBlock = 16;
	
	Mat imgErreur(h,w,CV_32SC1);
	Mat imgVecteur(h,w, CV_32SC2);
	Mat img_reconstruite(h, w, CV_32SC1);

	//Calcul du Temps d'execution
	TickMeter tm;
	tm.start();
	//Calcul de la carte d'erreur, de la carte de mouvement entre une image à l'instant t et à l'instant t+1. 0 pour SVA, 1 pour EQM et 2 pour EM
	//indiquez la taille de la fenêtre
	calculMouvement(vectorInput[0], vectorInput2[0],sizeBlock, 20, 2, imgErreur, imgVecteur);
	tm.stop();
	cout << "Temps :" << tm.getTimeMilli() << '\n';

	//Reconstruction de l'image à partir de la carte d'erreur et de mouvement
	reconstruction(img_reconstruite, vectorInput[0], imgErreur, imgVecteur, sizeBlock);

	Mat erreur(h,w,CV_8UC1);
	Mat img_reconstruite_UCV(h, w, CV_8UC1);
	// Conversion des images de float récupérées en images d'uchar
	imgErreur.convertTo(erreur,CV_8UC1);
	img_reconstruite.convertTo(img_reconstruite_UCV, CV_8UC1);
	
	//Coloriage de la carte d'erreur
	Mat erreur_color;
	applyColorMap(erreur, erreur_color, COLORMAP_JET);
	
	

	// https://docs.opencv.org/3.3.1/d7/d8b/tutorial_py_lucas_kanade.html
	// Calcul du flow de l'image
	Mat hsv(h,w, CV_8UC3);
	
	Mat flowx(h,w,CV_32FC1);
	Mat flowy(h,w,CV_32FC1);

	Mat imgVecteur_float;
	imgVecteur.convertTo(imgVecteur_float, CV_32FC2);
	
	//Sération des valeurs de la carte de mouvement selon les x et les y
	for(int i =0; i<h; i++){
		for(int j =0; j<w; j++){
			flowx.at<float>(i,j) = imgVecteur_float.at<Vec2f>(i,j)[0];
			flowy.at<float>(i,j) = imgVecteur_float.at<Vec2f>(i,j)[1];
		}
	}
		Mat mag;
		Mat ang;
		Mat magNorm;
		
		//Passage des coord cart to polar. Les valeurs sont stockées dans mag et dans ang
		cartToPolar(flowx,flowy,mag,ang,true);


		Mat convertMag;
		mag.convertTo(convertMag, CV_8UC1);
		Mat convertAng;
		ang.convertTo(convertAng, CV_8UC1);

		//Mag peut être négatif donc on normalise 
		normalize(convertMag, magNorm, 0, 255, NORM_MINMAX);
		
	for(int i =0; i<h; i++){
		for(int j =0; j<w; j++){
			hsv.at<Vec3b>(i, j)[0] = convertAng.at<uchar>(i, j);
			hsv.at<Vec3b>(i, j)[1] = (unsigned char) 255;
			hsv.at<Vec3b>(i, j)[2] = magNorm.at<uchar>(i, j);

		}
	}

	//Passage de HSV à BGR
	  Mat bgr;
	  cvtColor(hsv, bgr, COLOR_HSV2BGR);

	  //Calcul du PSNR entre l'image reconstruite et l'image 
	  double psnr_result = psnr(img_reconstruite_UCV, vectorInput2[0]);

	  //Calcul d'entropie
	  Mat histo_erreur;
	  computeHistogram(erreur, histo_erreur);
	  double entropie_erreur;
	  entropy(histo_erreur, vectorInput[0].rows*vectorInput[0].rows, entropie_erreur);

	//=======================================================================================
	// AFFICHAGE
	//=======================================================================================
	
	// Affichage des images
	imshow("imgErreur", erreur_color);
	imshow("BGR", bgr);
	imshow("imgretravaille", img_reconstruite_UCV);

		cout << "Valeur du psnr Y : " << psnr_result << " dB\n";
	cout << "-------------Prediction Competitive---------------\n";
	//cout << "Energie erreur prediction : " << energie_comp << "\n";
	cout << "Entropy erreur : " << entropie_erreur << "\n";
	cout << "NORME " << normMax(imgVecteur) << std::endl;

	//=======================================================================================
	// ECRITURE
	//=======================================================================================

	imwrite("imgErreur.jpg", erreur_color);
	imwrite("Bge.jpg", bgr);

	cvWaitKey();

	return 0;
}
