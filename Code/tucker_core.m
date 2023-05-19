function[F,U,V]=tucker_core(X,J1,J2,tol)

F=zeros(J1,J2,size(X,3));
U=rand(size(X,1),J1);
V=rand(size(X,2),J2);

S1_stari=zeros(size(X,1),1);
S1_novi=S1_stari;
S2_stari=zeros(size(X,2),1);
S2_novi=S2_stari;

kak_se_spusta=[];
iter=0;

poc=0;

while poc==0 || max(norm(S1_stari-S1_novi),norm(S2_stari-S2_novi))>tol
  if poc==0
    poc=1;
  endif

  S1_stari=S1_novi;
  B=ten_mat_mult(X,V',2);
  [u,s,v]=svd(unfold(B,1));
  U=u(:,1:J1);
  S1_novi=diag(s);

  S2_stari=S2_novi;
  B=ten_mat_mult(X,U',1);
  [u,s,v]=svd(unfold(B,2));
  V=u(:,1:J2);
  S2_novi=diag(s);

  iter=iter+1;
  kak_se_spusta(iter)=max(norm(S1_stari-S1_novi),norm(S2_stari-S2_novi));
end

F=ten_mat_mult(X,U',1);
F=ten_mat_mult(F,V',2);

#plot(kak_se_spusta)
#kak_se_spusta;
#iter;

end
