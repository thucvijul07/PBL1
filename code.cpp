#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<time.h>
// 4 neural dau vao
// 4 neural an
// 1 neural dau ra
double data[1001],cost[1001],loss[1001];
double lr = 0.1;
int order[1000],n,d;
double x[5],b[5],a[5],z[5];
double ao,bo,zo,y,min,max;		
double w[5][6];
int nn = 4,ni = 4,no = 1;
double dz[5],da[5],dw[5][6],db[5],dbo,dao,dzo;
char inputfile[50],weightfile[50] = "weight.out";

void out_result(double ans){
	printf("Du lieu du doan ngay tiep theo la:\n");
	printf("  %2d  |  %15.lf\n",n+1,ans);
}

void out_data(){
	
	printf(" Ngay |       Doanh thu        \n");
	
	for (int i=1;i<=n;i++){
		printf("  %2d  |  %15.lf\n",order[i],data[i]);
	}
}

double sigmoid(double t){
	return 1/(1+exp(-t));
}

void input_file(double data[], int order[], int &n, char tenFile[50]){		// dung de lay input tu file
	FILE *f;
	f = fopen(tenFile,"r");
	if(f==NULL){
		printf("\n Loi Mo File \n");
		return;
	}int i=1;
	while(!feof(f)){
	// lay input tu file
		fscanf(f,"%d",&order[i]);
		fscanf(f,"%lf",&data[i]);
		i++;	
	}fclose(f);
	n = i-2;
}

// dung de in ra w b bo cost ra file
void out_file(char tenFile[100]){
	FILE *f;
	f = fopen(tenFile, "a");
	if(f==NULL){
		printf("\n Loi mo file \n");
		return;
	}
	
	int i,j;
	for(i=1;i<=nn;i++){			// in ra file tu w[1][1] toi w[4][4]
		for(j=1;j<=nn;j++){
			fprintf(f,"%lf   ",w[i][j]);
		}fprintf(f,"%lf   ",b[i]);
	}for(i=1;i<=nn;i++){
		fprintf(f, "%lf   ",w[i][nn+1]);
	}fprintf(f,"%lf   \n",bo);
	fclose(f);
}

void rand_weight(double w[][6],double b[],double &bo, int nn){
	int i,j;
	srand((int)time(0));
	double r ;
	for(i=1;i<=nn;i++){
		for(j=1;j<=nn+1;j++){
			r = 10 + rand()%(99 - 10);
			r /= 100;
			w[i][j] = r;
		}
	}for(i=1;i<=nn;i++){
		r = 10 + rand()%(99 - 10);
		r /= 100;
		b[i] = r;
	}r = 10 + rand()%(99-10);
	r /= 100;
	bo = r;
}

void Normalize_Data(){
	int i;
	double max = data[1],min = data[1];
	for(i=1;i<=n;i++){
		if(data[i]>max){
			max = data[i];
		}if(data[i]<min){
			min = data[i];
		}
	}
	for(i=1;i<=n;i++){
		data[i] = 0.9*(data[i]-min)/(max-min) + 0.1;
	}
}

void proccess_data(double data[],int n){				
	double max = data[1];
	int i;
	d = 0;
	for(i=1;i<=n;i++){		
		if(data[i]>=max){
			max = data[i];
		}
	}
	while(max>=1){			
		max /= 10;
		d++;
	}
	for(i=1;i<=n;i++){
		data[i] /= pow(10,d);
	}
}

void take_input(int k,double x[],double data[],int ni){		// lay ra 4 input tu data
	int i,j=1;
	for(i=k;i<k+ni;i++){
		x[j]=data[i];
		j++;
	}y = data[k+4];
}
// forward progation process 
void forward_proga(){
		// tinh z va a
	double sum,pt ;
	int i,j;
	for(i=1;i<=nn;i++){
		sum = 0;
		for(j=1;j<=nn;j++){
			pt = w[j][i]*x[j];
			sum += pt;
		}z[i] = sum + b[i];
		a[i] = sigmoid(z[i]);
	}
		// tinh zo va ao
	sum = 0;
	for(i=1;i<=nn;i++){
		pt = w[i][nn+1]*a[i];
		sum += pt;
	}zo = sum + bo;
	ao = sigmoid(zo);
}

// update weight process (backprogation)
void update_weight(){
	int i,j;
	// ouput layer -> hidden layer
	dzo = ao - y;
	for(i=1;i<=nn;i++){
		w[i][nn+1] = w[i][nn+1] - a[i]*dzo;
		da[i] = w[i][nn+1]*dzo;
	}dbo = dzo;
	//hidden layer -> input layer
	for(i=1;i<=nn;i++){
		dz[i] = a[i]*(1-a[i])*da[i];
		db[i] = dz[i];
	}for(i=1;i<=nn;i++){
		for(j=1;j<=nn;j++){
			dw[i][j] = x[i]*dz[j];
		}
	}
	for(i=1;i<=nn;i++){
		for(j=1;j<=nn+1;j++){
			w[i][j] -= lr*dw[i][j];
		}
	}
	for(i=1;i<=nn;i++){
		b[i] -= lr*db[i];
	}bo -= lr*dbo;
}

// after progation use to calculate loss function and cost
void after_proga(int i){
	double sum = 0;
	cost[i] = -(y*log(ao)+(1-y)*log(1-ao));		
	for (int k = 1;k<=i;k++){					
		sum += cost[k];
	}loss[i] = sum/i;
}

// training process
void train(char tenFile[50]){
	int i,j;
	rand_weight(w,b,bo,nn);
	out_file(tenFile);
	for(i=1;i<=n-ni;i++){
		take_input(i,x,data,ni);
		forward_proga();
		update_weight();
		after_proga(i);
		out_file(tenFile);
	}
}

void proccess_1(){
		lr = lr * 10;
		printf("\n Nhap ten file du lieu: ");
		gets(inputfile);
		input_file(data,order,n,inputfile);
		out_data();
		proccess_data(data,n);
		train(weightfile);
}

void take_input_2(){
	int i;
	for(i=1;i<=ni;i++){
		x[i] = data[n-ni+i]*pow(10,d);
	}
}

void take_weight(char tenFile[50]){
	FILE *f;
	f = fopen(tenFile, "r");
	if(f==NULL){
		printf("\n Loi mo file \n");
		return;
	}
	int i,j;
	for(i=1;i<=nn;i++){		
		for(j=1;j<=nn;j++){
			fscanf(f,"%lf",&w[i][j]);
		}fscanf(f,"%lf",&b[i]);
	}for(i=1;i<=nn;i++){
		fscanf(f,"%lf",&w[i][nn+1]);
	}fscanf(f,"%lf",&bo);
	fclose(f);
}

void proccess_2(){
	take_input_2();
	proccess_data(x,ni);
	take_weight(weightfile);
	forward_proga();
	double ans = ao*pow(10,d);
	out_result(ans);
}

int main(){
	printf("Sinh vien: HUYNH THI THUC VI\n");
	printf("           VO THI HONG TIEN\n");
	printf("De tai: BAI TOAN DU DOAN DOANH SO BAN HANG SU DUNG MANG NEURAL\n");
	proccess_1();
	proccess_2();
	return 0;
}
