#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <TROOT.h>
#include <TChain.h>
#include <TFile.h>


using namespace std;

vector<vector<string>> read_csv( string filename = "prediction.csv" ){

    ifstream file( filename.c_str() );
    vector<vector<string>> data;
    string line;

    while ( getline(file, line) ){
        vector<string> row;
        stringstream ss(line);
        string cell;

        while (getline(ss,cell,',')){
            row.push_back(cell);
        }
        data.push_back(row);
    }

    return data; // data[0][:] = {label,prediction,weight,trigID,subrun}
}   


void Calc_score(string dirname){
    cout<<" dirname : " << dirname << endl; 
    //vector<vector<string>> sample = read_csv( dirname + "/prediction.csv" );
    vector<vector<string>> sample = read_csv( dirname + "/bdt__xgboost__depth_5__estimator_550__lr_0.1__random_18__test_size_0.250__noPedestal__JADEv0.csv" );
    cout<< sample[0][0]<<endl;
    cout << sample.size() <<endl;
    for (int i=0; i<sample[0].size() ; i++){
        cout << sample[0][i] <<endl;
    }

    vector<int> labels;
    vector<double> preds;
    vector<double> weights;

    for(int i=1; i< sample.size(); i++){
        labels.push_back( stoi(sample[i][2]) );
        preds.push_back( stod(sample[i][1]) );
        weights.push_back( stod(sample[i][3]) );
    }

    vector<double> pred_ME;
    vector<double> pred_FN;

    for(int i=1; i< sample.size(); i++){
        if( labels[i] == 1 ){ pred_ME.push_back(preds[i]); }
        if( labels[i] == 0 ){ pred_FN.push_back(preds[i]); }
    }
    sort(pred_ME.begin(), pred_ME.end());
    sort(pred_FN.begin(), pred_FN.end());
    //for(int i=0; i<pred_ME.size(); i++){  cout<<pred_ME[i]<<", ";}

    int fn_95 = 0.95*pred_FN.size();
    double rate_ME = 0.0;
    double rate_FN = 0.0;
    for(int i=0; i< pred_ME.size(); i++){ 
        if( pred_ME[i]>pred_FN[fn_95] ){ rate_ME += 1.0; }
    }
    for(int i=0; i< pred_FN.size(); i++){ 
        if( pred_FN[i]<pred_FN[fn_95] ){ rate_FN += 1.0; }
    }
    rate_ME /= double(pred_ME.size());
    rate_FN /= double(pred_FN.size());

    cout<< Form( "ME efficiency : %.3f +- %.3f %%",100*rate_ME,100*sqrt(rate_ME*(1-rate_ME)/pred_ME.size()) )<<endl;
    cout<< Form( "FN rejection : %.3f +- %.3f %%",100*rate_FN,100*sqrt(rate_FN*(1-rate_FN)/pred_FN.size()) )<<endl;

    TH2D *h_base = new TH2D("",";CNN score;Normalized",50,0,1,1000,0,1);
    TH1D *h_ME = new TH1D("ME","ME;score",50,0,1);
    TH1D *h_FN = new TH1D("FN","FN;score",50,0,1);

    for(int i=1; i< sample.size(); i++){
        if( labels[i] == 1 ){ h_ME->Fill(preds[i]); }
        if( labels[i] == 0 ){ h_FN->Fill(preds[i]); }
    }
     
    h_ME->Scale(1./h_ME->Integral());
    h_FN->Scale(1./h_FN->Integral());

    h_ME->SetLineWidth(2);
    h_FN->SetLineWidth(2);

    h_base->Draw();
    h_base->SetLineColor(0);
    h_ME->Draw("HIST same");
    h_ME->GetYaxis()->SetRangeUser(0,1);
    h_ME->SetLineColor(1);
    h_FN->Draw("HIST same");
    h_FN->SetLineColor(2);
    gPad->BuildLegend(0.40,0.70,0.60,0.85,"");
    gPad->SetGrid();
    gStyle->SetOptStat(0);
     
    cout<< "ME : "<<h_ME->Integral()<<endl;
    cout<< "FN : "<<h_FN->Integral()<<endl;

}
