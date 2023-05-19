TRAIN = load('ECG200_TRAIN.tsv');
TEST = load('ECG200_TEST.tsv');

%100 signala, dimenzija vremena 96
X_train= TRAIN(:, 2:end); %100x96
y_train= TRAIN(:,1); %100x1

%100 signala, dimenzija vremena 96
X_test= TEST(:, 2:end); %100x96
y_test= TEST(:,1); %100x96


%formatiranje TRAIN podataka
[N_tr, T_tr]=size(X_train);

trX=cell(N_tr,1);
for i=1:N_tr
  trX{i}=reshape(X_train(i,:),T_tr,1);
endfor

br_klasa=size(unique(y_train),1);

trY=zeros(br_klasa,N_tr*T_tr);
for i=1:N_tr
  if y_train(i)==-1
    trY(1, (i-1)*T_tr+1:i*T_tr)=1;
  else
    trY(2, (i-1)*T_tr+1:i*T_tr)=1;
  endif
endfor


%formatiranje TEST podataka
[N_ts, T_ts]=size(X_test);

tsX=cell(N_ts,1);
for i=1:N_ts
  tsX{i}=reshape(X_test(i,:),T_ts,1);
endfor

tsY=zeros(br_klasa,N_ts*T_ts);
for i=1:N_ts
  if y_test(i)==-1
    tsY(1, (i-1)*T_ts+1:i*T_ts)=1;
  else
    tsY(2, (i-1)*T_ts+1:i*T_ts)=1;
  endif
endfor


%ESN TRENIRANJE

washout=0;

N_esn=50; %broj unutarnjih cvorova u esn-u
esn = ESN(N_esn, 'leakRate', 0.3, 'spectralRadius', 0.5, 'regularization', 1e-8);

esn.train(trX, trY, washout);

internal_train=reshape(esn.internalState,N_esn,T_tr,N_tr);

output = esn.predict(tsX, washout);

internal_test=reshape(esn.internalState,N_esn,T_ts,N_ts);

[F,U,V]=tucker_core2(internal_train,25,4,1e-6);
#G=tucker_core(internal_test,25,4,1e-2);
G=ten_mat_mult(internal_test,U',1);
G=ten_mat_mult(G,V',2);

corr_tenz=0;
for i=1:N_ts
  M=G(:,:,i);
  index_min=1;
  min=norm(M-F(:,:,1));
  for j=2:N_tr
    if norm(M-F(:,:,j))<min
      index_min=j;
      min=norm(M-F(:,:,j));
    endif
  endfor
  if y_train(index_min)==y_test(i)
    corr_tenz=corr_tenz+1;
  endif
endfor

corr_weight=0;

for i=1:N_ts
  suma=zeros(1,br_klasa);
  for j=1:16
    suma=suma+output((i-1)*T_ts+j,:);
  endfor
  [~,indeks]=max(suma);
  if (indeks==1 && y_test(i)==-1) || (indeks==2 && y_test(i)==1)
    corr_weight=corr_weight+1;
  endif
endfor





%SVE ISTO ALI SAMO PROVJERAVA PSOTOTAK TOCNOSTI NA TRAINU
%{


tsX=trX;
tsY=trY;
N_ts=N_tr;
T_ts=T_tr;
output = esn.predict(tsX, washout);

internal_test=reshape(esn.internalState,N_esn,T_ts,N_ts);

[F,U,V]=tucker_core2(internal_train,25,4,1e-6);
#G=tucker_core(internal_test,25,4,1e-2);
G=ten_mat_mult(internal_test,U',1);
G=ten_mat_mult(G,V',2);

corr_tenz2=0;
for i=1:N_ts
  M=G(:,:,i);
  index_min=1;
  min=norm(M-F(:,:,1));
  for j=2:N_tr
    if norm(M-F(:,:,j))<min
      index_min=j;
      min=norm(M-F(:,:,j));
    endif
  endfor
  if y_train(index_min)==y_train(i)
    corr_tenz2=corr_tenz2+1;
  endif
endfor


corr_weight2=0;

for i=1:N_ts
  suma=zeros(1,br_klasa);
  for j=1:16
    suma=suma+output((i-1)*T_ts+j,:);
  endfor
  [~,indeks]=max(suma);
  if (indeks==1 && y_train(i)==-1) || (indeks==2 && y_train(i)==1)
    corr_weight2=corr_weight2+1;
  endif

endfor



%}
