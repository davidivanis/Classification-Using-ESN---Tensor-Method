load azip
load dzip
load testzip
load dtest

trX=cell(1707,1);
for i=1:1707
  trX{i}=reshape(azip(:,i),16,16);
end

trY=zeros(10,16*1707);
for i=1:1707
  for j=0:9
    if j==dzip(i)
      trY(j+1,(i-1)*16+1:i*16)=1;
    endif
  endfor
end

tsX=cell(2007,1);
for i=1:2007
  tsX{i}=reshape(testzip(:,i),16,16);
end

tsY=zeros(10,16*2007);
for i=1:2007
  for j=0:9
    if j==dtest(i)
      tsY(j+1,(i-1)*16+1:i*16)=1;
    endif
  endfor
end

washout=0;
Nr=10;

esn = ESN(Nr, 'leakRate', 0.3, 'spectralRadius', 0.999, 'regularization', 1e-8);

esn.train(trX, trY, washout);

internal_train=reshape(esn.internalState,Nr,16,1707);

output = esn.predict(tsX, washout);

internal_test=reshape(esn.internalState,Nr,16,2007);

[F,U,V]=tucker_core(internal_train,5,4,1e-6);
G=ten_mat_mult(internal_test,U',1);
G=ten_mat_mult(G,V',2);

corr_tenz=0;
for i=1:2007
  M=G(:,:,i);
  index_min=1;
  min=norm(M-F(:,:,1));
  for j=2:1707
    if norm(M-F(:,:,j))<min
      index_min=j;
      min=norm(M-F(:,:,j));
    endif
  end
  if dzip(index_min)==dtest(i)
    corr_tenz=corr_tenz+1;
  endif
end

corr_weight=0;

for i=1:2007
  suma=zeros(1,10);
  for j=1:16
    suma=suma+output((i-1)*16+j,:);
  end
  [~,indeks]=max(suma);
  if indeks==dtest(i)+1
    corr_weight=corr_weight+1;
  end
end
