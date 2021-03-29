
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
// quantification
//=======================================================================================
float quantification(float input, int step)
{
	// Si le pas de quantification est de 0 alors nous n'effectuons pas de quantification
	if (step == 0) return input;
	// Autrement on quantifie la valeur avec un pas = step fournit en paramètre
	else {
		return (step * (int)(input / step + 0.5));
	}
}

//=======================================================================================
// quantification inverse
//=======================================================================================
float quantification_inverse(float input, int step)
{
	// Si le pas de quantification est de 0 alors nous n'effectuons pas de quantification inverse
	if (step == 0) return input;
	// Quantification inverse n'effectuant rien dans notre cas
	else {
		return ((input / step) * step);
	}
}


//=======================================================================================
// codeur
//=======================================================================================
void codeur(Mat & input, Mat & output, Mat & erreur_pred, int pred, int step_quantif) {

	// Déclaration de variable permettant de stocker les valeurs à réutiliser dans la boucle fermée du codeur
	int X;
	int Eestim;
	int E;
	int Xpred;

	// Création d'une image intermédiaire qui nous permettera de stocker les valeurs de sortie de l'additionneur entre la quantif inverse de la valeur
	// et Xpred. Cette image est l'équivalent de l'image de sortie du décodeur qui est utilisé dans son prédicteur et donc c'est sur cette image que la prédiction se fera
	// afin d'obtenir la meme procedure que sur le décodeur.
	Mat Xestim(input.rows, input.cols, CV_32FC1);

	// On parcours l'image d'entrée sur chaque pixel
	for (int i = 0; i < input.rows; i++) {
		for (int j = 0; j < input.cols; j++) {

			// La valeur d'entrée du codeur (pixel de l'image)
			X = input.at<float>(i, j);
			
			//1 : Calcul Xpred en fonction du prédicteur utilisé
			switch (pred)
			{
				// Predicteur MICD mono-dimensionnelle
				case 0:
					if (j - 1 < 0) Xpred = 128;
					else Xpred = Xestim.at<float>(i, j - 1);

					break;
				// Predicteur MICD bi-dimensionnelle
				case 1:
					if (j - 1 < 0 || i - 1 < 0) Xpred = 128;
					else Xpred = (Xestim.at<float>(i, j - 1) + Xestim.at<float>(i - 1, j)) / 2;

					break;
				// Predicteur MICDA
				default:
					if (j - 1 < 0 || i - 1 < 0) Xpred = 128;
					else if (abs(Xestim.at<float>(i - 1, j) - Xestim.at<float>(i - 1, j - 1)) < abs(Xestim.at<float>(i, j - 1) - Xestim.at<float>(i - 1, j - 1)))
					{
						Xpred = Xestim.at<float>(i, j - 1);
					}
					else Xpred = Xestim.at<float>(i - 1, j);
			}
			//2 : Calcul E(i,j)
			E = X - Xpred;

			//Enregistrement de la valeur de l'erreur de prédiction dans l'image d'erreur
			erreur_pred.at<float>(i, j) = E;

			//3 : Quantification
			output.at<float>(i, j) = quantification(E, step_quantif);

			//4 : Quantification inverse
			Eestim = quantification_inverse(output.at<float>(i, j), step_quantif);

			//5 : Calcul Xestim(i,j)
			Xestim.at<float>(i, j) = Eestim + Xpred;
		}
	}
}

//=======================================================================================
// decodeur
//=======================================================================================
void decodeur(Mat & input, Mat & output,int pred, int step_quantif) {

	// Déclaration de variable permettant de stocker les valeurs à réutiliser
	int Eestim;
	int Xpred;

	// On parcours l'image d'entrée sur chaque pixel
	for (int i = 0; i < input.rows; i++) {
		for (int j = 0; j < input.cols; j++) {

			//1 : Quantification inverse
			Eestim = quantification_inverse(input.at<float>(i, j), step_quantif);

			//2 : Calcul Xpred en fonction du predicteur
			switch (pred)
			{
				// Predicteur MICD mono-dimensionnelle
				case 0:
					if (j - 1 < 0) Xpred = 128;
					else Xpred = output.at<float>(i, j - 1);

					break;
				// Predicteur MICD bi-dimensionnelle
				case 1:
					if (j - 1 < 0 || i - 1 < 0) Xpred = 128;
					else Xpred = (output.at<float>(i, j - 1) + output.at<float>(i - 1, j)) / 2;

					break;
				// Predicteur MICDA
				default:
					if (j - 1 < 0 || i - 1 < 0) Xpred = 128;
					else if (abs(output.at<float>(i - 1, j) - output.at<float>(i - 1, j - 1)) < abs(output.at<float>(i, j - 1) - output.at<float>(i - 1, j - 1)))
					{
						Xpred = output.at<float>(i, j - 1);
					}
					else Xpred = output.at<float>(i - 1, j);
			}

			//3 : Calcul sortie
			output.at<float>(i, j) = Eestim + Xpred;
		}
	}
}

//=======================================================================================
// transferImage
// Fonction permettant de simuler le transfert d'une image avec le prédicteur 0,1 ou 2 et une quantif = step_quantif
//=======================================================================================
void transferImage(Mat & input, Mat & output, Mat & erreur_pred, int pred, int step_quantif){
	
	// Création d'une image intermédaire qui contiendra l'image de sortie du codeur (contenant les erreurs de prédiction (quantifiés si step_quantif != 0))
	Mat inter(input.rows, input.cols, CV_32FC1);

	// Appel de la fonction codeur afin d'obtenir l'image de sortie du codeur ainsi que l'image d'erreur de prédiction
	// L'image d'erreur de prédiciton et de sortie de codeur seront identique dans le cas où il n'y a pas de quantification
	codeur(input, inter,erreur_pred, pred, step_quantif);

	// Appel de la fonction décodeur afin d'obtenir l'image décodé avec l'image se sortie du codeut en paramètre
	decodeur(inter, output, pred, step_quantif);
}

//=======================================================================================
// codeur
//=======================================================================================
void codeurCompetition(Mat & input, Mat & output, Mat & erreur_pred, int step_quantif) {

	// Déclaration de variable permettant de stocker les valeurs à réutiliser dans la boucle fermée du codeur
	int X;
	int Eestim;
	int E;
	int Xpred, Xpred0, Xpred1, Xpred2;
	float erreur_inter;

	// Création d'une image intermédiaire qui nous permettera de stocker les valeurs de sortie de l'additionneur entre la quantif inverse de la valeur
	// et Xpred. Cette image est l'équivalent de l'image de sortie du décodeur qui est utilisé dans son prédicteur et donc c'est sur cette image que la prédiction se fera
	// afin d'obtenir la meme procedure que sur le décodeur.
	Mat Xestim(input.rows, input.cols, CV_32FC1);

	// Parcours de chaque pixel de l'image
	for (int i = 0; i < input.rows; i++) {
		for (int j = 0; j < input.cols; j++) {

			// La valeur d'entrée du codeur (pixel de l'image)
			X = input.at<float>(i, j);

			//1 : Calcul Xpred pour les 3 prédicteurs différents Xpred0, Xpred1 et Xpred2
			// Predicteur MICD mono-dimensionnelle
			if (j - 1 < 0) Xpred0 = 128;
			else Xpred0 = Xestim.at<float>(i, j - 1);

			// Predicteur MICD bi-dimensionnelle
			if (j - 1 < 0 || i - 1 < 0) Xpred1 = 128;
			else Xpred1 = (Xestim.at<float>(i, j - 1) + Xestim.at<float>(i - 1, j)) / 2;

			// Predicteur MICDA
			if (j - 1 < 0 || i - 1 < 0) Xpred2 = 128;
			else if (abs(Xestim.at<float>(i - 1, j) - Xestim.at<float>(i - 1, j - 1)) < abs(Xestim.at<float>(i, j - 1) - Xestim.at<float>(i - 1, j - 1)))
			{
				Xpred2 = Xestim.at<float>(i, j - 1);
			}
			else Xpred2 = Xestim.at<float>(i - 1, j);

			//2 : Calcul E(i,j)
			// E(i,j) est sélectionné en prenant la valeur minimale entre les 3 erreurs des prédictions précédentes
			E = X - Xpred0;
			Xpred = Xpred0;

			erreur_inter = X - Xpred1;
			if (erreur_inter < E)
			{
				E = erreur_inter;
				Xpred = Xpred1;
			}

			erreur_inter = X - Xpred2;
			if (erreur_inter < E)
			{
				E = erreur_inter;
				Xpred = Xpred2;
			}

			//Erreur de prédiction
			erreur_pred.at<float>(i, j) = E;

			//3 : Quantification
			output.at<float>(i, j) = quantification(E, step_quantif);

			//4 : Quantification inverse
			Eestim = quantification_inverse(output.at<float>(i, j), step_quantif);

			//5 : Calcul Xestim(i,j)
			Xestim.at<float>(i, j) = Eestim + Xpred;
		}
	}
}

//=======================================================================================
//=======================================================================================
// MAIN
//=======================================================================================
//=======================================================================================
int main(int argc, char** argv){
	/*if (argc < 3){
	std::cout << "No image data... At least two argument is required! \n";
	return -1;
	}
  
	Mat inputImageSrc; 

	// Ouvrir l'image d'entrée et vérifier que l'ouverture du fichier se déroule normalement
	inputImageSrc = imread(argv[1], CV_LOAD_IMAGE_COLOR);
	if(!inputImageSrc.data ) { // Check for invalid input
		std::cout <<  "Could not open or find the image " << argv[1] << std::endl ;
		waitKey(0); // Wait for a keystroke in the window
		return -1;
	}
*/

	// Lecture des 2 fichiers sans passer par les arguments fournis
	Mat inputImageSrc = imread("../../boats.bmp", CV_LOAD_IMAGE_COLOR);
	/*Mat inputImageSrc2 = imread("../../smileyDegQ2.jpg", CV_LOAD_IMAGE_COLOR);*/

	// Images qui détiendront les images en YCrCb
	Mat convertImageSrc;
  	
	// Conversion des 2 images RGB en image YCrCb
	cvtColor(inputImageSrc, convertImageSrc, COLOR_BGR2YCrCb);
  
	// Vecteurs d'image qui détiendront les valeurs Y Cr et Cb des 2 images
	vector<Mat> vectorInput;
  
	// Découpage des 3 composantes Y Cr et Cb dans des vecteurs d'image des 2 images
	split(inputImageSrc, vectorInput);
	// On choisit de travailler avec la composante Y de l'image boats.bmp

	// Choix du prédicteur :
	int pred;
	cout << "Predicteur : \n 0 : MICD mono-dimensionnelle  \n 1 : MICD bi-dimensionnelle  \n 2 : MICDA \n";
	cin >> pred;
	
	// Conversion de l'image en image de float dans input_convert
	Mat input_convert;
	vectorInput[0].convertTo(input_convert, CV_32FC1);
	
	// Déclaration de l'image de float de sortie du décodeur
	Mat output_float(vectorInput[0].rows, vectorInput[0].cols, CV_32FC1);
	// Déclaration de l'image de float des erreurs de prédiction du codeur
	Mat erreur_pred_float(vectorInput[0].rows, vectorInput[0].cols, CV_32FC1);

	// On définit un pas de quantification, le mettre à 0 si aucune quantification
	int step_quantif = 4;
	// Appel d la fonction de codage, décodage de l'image
	transferImage(input_convert, output_float, erreur_pred_float, pred, step_quantif);
	
	Mat output;
	Mat erreur_pred;
	// Conversion des images de float récupérées en images d'uchar
	output_float.convertTo(output,CV_8UC1);
	erreur_pred_float.convertTo(erreur_pred, CV_8UC1);

	Mat histo_erreur_pred;
	Mat histo_img_ori;
	Mat histo_img_sortie;
	// Création de 3 histogrammes d'images: 
	// Histogramme des erreurs de prédiction
	computeHistogram(erreur_pred, histo_erreur_pred);
	// Histogramme de l'image originale
	computeHistogram(vectorInput[0], histo_img_ori);
	// Histogramme des l'image de sortie
	computeHistogram(output, histo_img_sortie);

	double energie_erreur, entropie_erreur, entropie_ori, entropie_sortie;
	// Récupération de l'énergie de l'image d'erreur
	energy(erreur_pred, energie_erreur);
	// Calcul des entropies des différents histogrammes
	entropy(histo_erreur_pred, vectorInput[0].rows*vectorInput[0].rows, entropie_erreur);
	entropy(histo_img_ori, vectorInput[0].rows*vectorInput[0].rows, entropie_ori);
	entropy(histo_img_sortie, vectorInput[0].rows*vectorInput[0].rows, entropie_sortie);


	//=======================================================================================
	// Calcul erreur prediction compétition
	//=======================================================================================

	// Déclaration de l'image de float de sortie du codeur compétitif (erreurs quantifiés)
	Mat output_comp(vectorInput[0].rows, vectorInput[0].cols, CV_32FC1);
	// Déclaration de l'image de float des erreurs de prédiction du codeur compétitif
	Mat erreur_pred_comp_float(vectorInput[0].rows, vectorInput[0].cols, CV_32FC1);
	// Appel de la fonction de codage de l'image
	codeurCompetition(input_convert, output_comp, erreur_pred_comp_float, step_quantif);

	Mat erreur_pred_comp;
	// Convertion de l'image des erreurs en uchar
	erreur_pred_comp_float.convertTo(erreur_pred_comp, CV_8UC1);
	
	Mat histo_erreur_pred_comp;
	// Création de l'histogramme des erreurs
	computeHistogram(erreur_pred_comp, histo_erreur_pred_comp);

	double energie_comp, entropie_comp;
	// Calcul de l'énergie et de l'entropy de l'image d'erreur
	energy(erreur_pred_comp, energie_comp);
	entropy(histo_erreur_pred_comp, vectorInput[0].rows*vectorInput[0].rows, entropie_comp);
	

	//=======================================================================================
	// AFFICHAGE
	//=======================================================================================

	// Affichage des images
	imshow("ImageAvantCodage", vectorInput[0]);
	imshow("ImageApresDecodage", output);
	imshow("Erreur de prediction", erreur_pred);
	imshow("Erreur de prediction competition", erreur_pred_comp);

	// Affichage de l'histogramme
	displayHistogram(histo_erreur_pred);

	// Affichage des données
	cout << "Energie erreur prediction : " << energie_erreur << "\n";
	cout << "Entropy erreur prediciton : " << entropie_erreur << "\n";
	cout << "Entropy image originale : " << entropie_ori << "\n";
	cout << "Entropy image decodee : " << entropie_sortie << "\n";
	// Affichage de la valeur du psnr de l'image decodee par rapport a l'image originale
	double psnr_result = psnr(output, vectorInput[0]);
	if (psnr_result != NULL)
	{
		cout << "Valeur du psnr Y : " << psnr_result << " dB\n";
	}
	cout << "-------------Prediction Competitive---------------\n";
	cout << "Energie erreur prediction : " << energie_comp << "\n";
	cout << "Entropy erreur prediciton : " << entropie_comp << "\n";

	//=======================================================================================
	// ECRITURE
	//=======================================================================================

	//imwrite("erreur_pred.jpg", erreur_pred);
	//imwrite("img_sortie.jpg", result);
	//imwrite("erreur_pred_comp.jpg", erreur_pred_comp);

	// Ecriture des données dans un fichier texte
	/*std::ofstream outfile("data.txt");

	outfile << "Energie erreur prediction : " << energie_erreur << "\n"
	<< "Entropy erreur prediciton : " << entropie_erreur << "\n"
	<< "Entropy image originale : " << entropie_ori << "\n"
	<< "Entropy image decodee : " << entropie_sortie << "\n"
	<< "Valeur du psnr Y : " << psnr_result << " dB\n"
	<< std::endl;

	outfile.close();*/

	cvWaitKey();

	return 0;
}
