#define vtkRenderingCore_AUTOINIT 4(vtkInteractionStyle,vtkRenderingFreeType,vtkRenderingFreeTypeOpenGL,vtkRenderingOpenGL)
#define vtkRenderingVolume_AUTOINIT 1(vtkRenderingVolumeOpenGL)
#include <vtkPolyDataReader.h>
#include <vtkPolyDataWriter.h>
#include <vtkPolyData.h>
#include <vtkSmartPointer.h>
#include <vtkPoints.h>
#include <vtkPolyDataMapper.h>
#include <vtkActor.h>
#include <vtkRenderWindow.h>
#include <vtkRenderer.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkProperty.h>
#include <vtkConeSource.h>
#include <vtkSphereSource.h>
#include <vtkCubeSource.h>
#include <vtkDiskSource.h>
#include <vtkCylinderSource.h>
#include <vtkArrowSource.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;


vtkSmartPointer<vtkPolyDataReader> readerValue (string filename){
	vtkSmartPointer<vtkPolyDataReader> reader =
	vtkSmartPointer<vtkPolyDataReader>::New();
	reader->SetFileName(filename.c_str());
	reader->Update();
	vtkPolyData* fileData = reader->GetOutput();

	vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New();
	vtkSmartPointer<vtkPolyData> polydataResult = vtkSmartPointer<vtkPolyData>::New();
	int nOfPnts = fileData->GetNumberOfPoints();

	//There are 5090 points
	int count = nOfPnts;
	//
	int arrayDim = 3; //initial dimension has set to 3 (x, y, z) coordinate

	//vector declaration
	vector<vector<double> >arrayPts(count, vector<double>(arrayDim));

	//loop through the points (row)
	//note we need to start with i=0 is because array start with index 0.
	for (vtkIdType i = 0; i < nOfPnts; i++)
	{
		double p[3];
		fileData->GetPoint(i, p);
		//cout  << p[0] << " " << p[1] << " " <<
		//p[2]<<","<<endl;

		//loop through the points and store into array (column)
		for (int j = 0; j <= 2; j++){

		arrayPts[i][j] = p[j];
		//cout << arrayPts[i][j] << " ";
		}
	}
	return reader;
}

int main(int argc, char* argv[])
{
	// Parse command line arguments
	

	string trainingf = argv[1];
	string trainingf2 = argv[2];
	string trainingf3 = argv[3];
	string testingf = argv[4];

	// Create a render window
	vtkRenderWindow* renWin = vtkRenderWindow::New();
	renWin->SetSize( 800, 600);
	//renWin->SetAAFrames(6);

	// Create an interactor
	vtkRenderWindowInteractor* iren = vtkRenderWindowInteractor::New();
	renWin->SetInteractor( iren );

	// Create a renderer
	const int NR = 2 ;
	const int NC = 3 ;
	const int N = NR*NC ;
	vtkRenderer* ren[N] ;
	for (int i=0; i<NR; i++)
		for (int j=0; j< NC; j++)
		{
			ren[i*NC+j] = vtkRenderer::New();
			ren[i*NC+j]->SetBackground(.3, .6, .3);
			
			ren[i*NC+j]->SetViewport( 				
				double(j)/double(NC),
				double(i)/double(NR), 				
				double(j+1)/double(NC),
				double(i+1)/double(NR));
			renWin->AddRenderer(ren[i*NC+j]);
		}

	{
		// trainingface1
		vtkSmartPointer<vtkPolyDataMapper> mapper =
		vtkSmartPointer<vtkPolyDataMapper>::New();
		mapper->SetInputConnection(readerValue(trainingf)->GetOutputPort());
		vtkActor* actor = vtkActor::New();
		actor->SetMapper(mapper);
		actor->GetProperty()->SetColor(0.0,1.0,1.0);
		ren[0]->AddActor(actor);
	}
	{
		// trainingface2
		vtkSmartPointer<vtkPolyDataMapper> mapper =
		vtkSmartPointer<vtkPolyDataMapper>::New();
		mapper->SetInputConnection(readerValue(trainingf2)->GetOutputPort());
		vtkActor* actor = vtkActor::New();
		actor->SetMapper(mapper);
		actor->GetProperty()->SetColor(0.0,1.0,1.0);
		ren[1]->AddActor(actor);
	}
	{
		// trainingface3
		vtkSmartPointer<vtkPolyDataMapper> mapper =
		vtkSmartPointer<vtkPolyDataMapper>::New();
		mapper->SetInputConnection(readerValue(trainingf3)->GetOutputPort());
		vtkActor* actor = vtkActor::New();
		actor->SetMapper(mapper);
		actor->GetProperty()->SetColor(0.0,1.0,1.0);
		ren[2]->AddActor(actor);
	}
	{
		// testingface
		vtkSmartPointer<vtkPolyDataMapper> mapper =
		vtkSmartPointer<vtkPolyDataMapper>::New();
		mapper->SetInputConnection(readerValue(testingf)->GetOutputPort());
		vtkActor* actor = vtkActor::New();
		actor->SetMapper(mapper);
		ren[3]->AddActor(actor);
	}



	// Initialize and enter interactive mode
	iren->Initialize();
	iren->Start();

	return 0 ;
}