%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Source separation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% I. Generation of an artificial mix of two artificial sources

%% Question 1
% 'Generate two random signals'

% use of the generator kernel random 'seed'
rand('seed', 1);

sig1 = rand(1,1000);
sig2 = rand(1,1000);


%% Question 2
% 'Process the signals to center and reduce the random signals (unitarian power)' 

sig1_c = (sig1 - mean(sig1)) ;
sig1_cr = sig1_c / sqrt(mean(sig1_c .* sig1_c ));
sig2_c = (sig2 - mean(sig2)) ;
sig2_cr = sig2_c / sqrt(mean(sig2_c .* sig2_c ));

x = 1:1000;
figure(1)
plot(x,sig1_cr) 
figure(2)
plot(x,sig2_cr) 


%% Question 3
% 'Display the correlation coefficient between the two sources and
% interprate'

corr_s = corrcoef(sig1_cr,sig2_cr);
% the correlation coefficient is the value out of the diagonal matrix
% corr_s.
% The coefficient is very low (0.0115) : it means that the two random
% signals are not correlated

% reminder: corr = (esp_12-esp_1*esp_2)/standev1/standev2;


%% Question 4
% 'Analyse s1 and s2 independance according to the bidimensional cloud of
% points of the signals'

figure(3)
title('bidimensional cloud of points')
xlabel('s1(t)')
ylabel('s2(t)')
scatter(sig1_cr, sig2_cr)

% The plot shows that s2 (respectively s1) is free to be defined if s1
% (respectively s2) has already been set.
% Signals are thus independant (so, not correlated)


%% Question 5
% 'Mix the signals with the matrix A=[1, 0.7; 0.8, 1] to get the observation vector X=[x1(t); x2(t)] 

A = [1, 0.7; 0.8, 1];
S = [sig1_cr; sig2_cr];
X = A * S;


%% Question 6
% 'Analyse x1 and x2 independance according to the bidimensional cloud of
% points of the signals'

corr_x = corrcoef(X(1,:) , X(2,:));
% The coefficient is very high (0.9605) : it means that the two random
% signals are correlated

figure(4)
scatter(X(1,:), X(2,:));

% The plot shows that x2 (respectively x1) is not free to be defined if x1
% (respectively x2) has already been set (if x1 is set to 0.5, x2 can only
% belong to the interval [0.4,1.1]. This restriction is a testimony of
% non-independance
% Signals are thus not independant


%% II. Bleaching

%% Question 1
% 'Compute the bleaching matrix in using the covariance matrix of X

C = cov(X');
[V,D] = eig(C);

% M = D^(-1/2)*V'
SQ_D = [1/sqrt(D(1,1)), 0; 0, 1/sqrt(D(2,2))];
M = SQ_D*V'; % because V is already an orthogonal matrix
Z = M*X;


%% Question 2
% 'Analyse z1 and z2 independance and correlation according to the bidimensional cloud of
% points of the signals'

corr_z = corrcoef(Z(1,:), Z(2,:));
% The coefficient is near 0 (10^-15) : it means that the two random
% signals are not correlated

figure(5)
scatter(Z(1,:), Z(2,:));
% The plot shows that z2 (respectively z1) is not free to be defined if z1
% (respectively z2) has already been set (if s1 is set to 0.5, z2 can only
% belong to the interval [-1.8,1.9]. This restriction is a testimony of
% non-independance
% Signals are thus not independant


%% III. Sources estimation

%% Question 1
% 'Reminds the rule in order to update w in the fulcrum algorithm'

% There are two steps in this algorithm: 
% 1) update of w: w = E{z(w'z)^3}-3w.||w||^2
% 2) normalization of w: w = w/||w||
% If w is not normalized at each step, it can impeach the algorithm to
% converge; it is compulsory to compare "same-normed objects" to judge the convergence 


%% Question 2
% 'write an iterative algorithm which optimisez w with the updating pattern
% of the question 1'

rand('seed', 2);
z1 = Z(1,:);
z2 = Z(2,:);
p=1;
n = 50;

% criteria of convergence
epsilon = 0.000001;

stock_w1 = zeros(2,n);
stock_y1 = zeros(n,1000);
stock_kurt_y1 = zeros(1,n);

w1_init = randn(2,1);

% initialization of the two first w before launching the while structure
w1 = w1_init;
w1_next = mean(Z.*(w1'*Z).^3,2) - 3*w1;
w1_next = w1_next/sqrt(w1_next'*w1_next);
y1 = w1'*Z;
y1_next = w1_next'*Z;
kurt_y1 = kurtosis(y1);
kurt_y1_next = kurtosis(y1_next);

% initialization of the the stocking matrices
stock_w1(:,1:2) = [w1, w1_next];
stock_y1(1:2,:) = [y1; y1_next];
stock_kurt_y1(1,1:2) = [kurt_y1, kurt_y1_next];

while abs(stock_kurt_y1(p+1) - stock_kurt_y1(p)) > epsilon

    p = p+1;
    w1 = stock_w1(:,p);
    w1_next = mean(Z.*(w1'*Z).^3,2) - 3*w1;
    w1_next = w1_next/sqrt(w1_next'*w1_next);
    y1_next = w1_next'*Z;
    kurt_y1_next = kurtosis(y1_next);
    
    stock_w1(:,p+1) = w1_next;
    stock_y1(p+1,:) = y1_next;
    stock_kurt_y1(p+1) = kurt_y1_next;
    
end

p=p+1;

%% Question 3
% 'Plot w1 components and kurt_y1'

figure(6)
plot(1:n,stock_w1(1,:),1:n,stock_w1(2,:))

figure(7)
plot(1:n,stock_kurt_y1)

% The convergence is considered as correct for epsilon = 10^-6 from 8
% iterations (all the variables are 0 afterwards)


%% Question 4
% 'Plot and compare y1 with s1 and s2'

y1 = stock_y1(p,:);

% Because of the indeterminations of permutations and signs, we plot here 4
% variables to emphasize the corresponding sources and estimations
figure(8)
plot(1:1000,y1-sig1_cr)
figure(9);
plot(1:1000,y1-sig2_cr)

figure(10);
plot(1:1000,y1+sig1_cr)
figure(11);
plot(1:1000,y1+sig2_cr)

% The plots are difficult to interpret but the extrema of y1+s2
% is substantially lower than those of s1 and s2. It means that y1
% is a pretty good estimator for -s2


%% Question 5
% 'Why w1 and w2 are orthonormal?'

% y1 and y2 are supposed to be independant; it means that E((y1;y2).(y1;y2)') is
% supposed to be equalled to Identity (or at least diagonal).
% thus, E(y1.y2) = 0. So, y1 and y2 are orthogonal.
% Also, z!=0. So if y1 and y2 are orthogonal, w1 and w2 are so too.
% And it is obvious that norms of w1 and w2 are 1 by the building rule of
% the algorithm.
% Finally, w1 and w2 are orthonormal.


%% Question 6
% 'Show that w1 and w2 are orthonormal'

% We have easily w1'*w2 = 0 and w1'*w1 = w2'*w2 = 1.


%% Question 7
% 'Infer y2 and compare y2 with s1 and s2'

alpha = atan(-stock_w1(1,p)/stock_w1(2,p));
w2 = [cos(alpha); sin(alpha)];
y2 = w2'*Z;

figure(12)
plot(1:1000,y2-sig1_cr)
figure(13);
plot(1:1000,y2-sig2_cr)

figure(14);
plot(1:1000,y2+sig1_cr)
figure(15);
plot(1:1000,y2+sig2_cr)

% y2 is very similar with -sig1_cr. It means that the separation seems
% accurate.
% Finally: y1 stands for estimator of -s2 and y2 for estimator of -s1


%% Question 8
% 'Analyse y1 and y2 independance and correlation according to the bidimensional cloud of
% points of the signals'

corr_y = corrcoef(y1,y2);
figure(10)
scatter(y1,y2);
% The signals y1 and y2 are effectively independant. It is what we wished.


%% IV. Measures of performance of the algorithms

%% Question 1
% 'Reminds scale factors and permutation indeterminations'

% Scale factors and permutation indeterminations are mainly dependent on constraints
% imposed to signal sources, the mixing nature and the algorithm
% implemented
% To fix scalefactors, standart solutions are :
% _ to adjust signal magnitudes in order to have output signals with a
% power of 1.
% _ to restore output magnitudes same as sources magnitudes
% Finally, only indeterminations of permutations and signs remains. 
% In the case of two sources, this constraint is not too difficult to
% manage.


%% Question 2
% 'Compute the RSI criteria to analyze the success of the algorithm'  

figure(11)
scatter(y1 ,-sig2_cr);

figure(12)
scatter(y2 , -sig1_cr);

rsi = 5*log10(mean(sig1_cr.^2,2)/mean((sig1_cr+y2).^2,2)) + 5*log10(mean(sig2_cr.^2,2)/mean((sig2_cr+y1).^2,2));

% RSI = 39.71


%% V. Test avec d'autres signaux

%% Question 1: 
% 'Sounds separation'

son1 = double(audioread('audio_mix1.wav'))';
son2 = double(audioread('audio_mix2.wav'))';

% Centered and reduced signals
son1_c = (son1-mean(son1));
son2_c = (son2-mean(son2));
son1_cr = son1_c / sqrt(mean(son1_c.^2));
son2_cr = son2_c / sqrt(mean(son2_c.^2));

X = [son1_cr; son2_cr];

% Bleaching
C = cov(X');
[V,D] = eig(C);
SQ_D = [1/sqrt(D(1,1)), 0; 0, 1/sqrt(D(2,2))];
M = SQ_D*V'; % because V is already an orthogonal matrix
Z = M*X;

% FastICA algorithm implementation
epsilon = 0.000001;
p=1;

stock_w1 = zeros(2,n);
stock_y1 = zeros(n,44100);
stock_kurt_y1 = zeros(1,n);

w1_init = randn(2,1);

w1 = w1_init;
w1_next = mean(Z.*(w1'*Z).^3,2) - 3*w1;
w1_next = w1_next/sqrt(w1_next'*w1_next);
y1 = w1'*Z;
y1_next = w1_next'*Z;
kurt_y1 = kurtosis(y1);
kurt_y1_next = kurtosis(y1_next);

stock_w1(:,1:2) = [w1, w1_next];
stock_y1(1:2,:) = [y1; y1_next];
stock_kurt_y1(1,1:2) = [kurt_y1, kurt_y1_next];

while abs(stock_kurt_y1(p+1) - stock_kurt_y1(p)) > epsilon

    p = p+1;
    w1 = stock_w1(:,p);
    w1_next = mean(Z.*(w1'*Z).^3,2) - 3*w1;
    w1_next = w1_next/sqrt(w1_next'*w1_next);
    y1_next = w1_next'*Z;
    kurt_y1_next = kurtosis(y1_next);
    
    stock_w1(:,p+1) = w1_next;
    stock_y1(p+1,:) = y1_next;
    stock_kurt_y1(p+1) = kurt_y1_next;
    
end

p=p+1;

% plots of the sounds 1 and 2
x = 1:44100;
figure(20)
plot(x,son1_cr) 
figure(21)
plot(x,son2_cr-son1_cr) 

figure(22)
scatter(Z(1,:), Z(2,:));

% w1 components and kurt_y1
figure(1)
plot(1:n,stock_w1(1,:),1:n,stock_w1(2,:))

figure(2)
plot(1:n,stock_kurt_y1)

% Because of the indeterminations of permutations and signs, we plot here 4
% variables to emphasize the corresponding sources and estimations 
y1 = stock_y1(p,:);

figure(3)
plot(1:44100,y1-son1_cr)
figure(4);
plot(1:44100,y1-son2_cr)

figure(5);
plot(1:44100,y1+son1_cr)
figure(6);
plot(1:44100,y1+son2_cr)

% same process for y2
alpha = atan(-stock_w1(1,p)/stock_w1(2,p));
w2 = [cos(alpha); sin(alpha)];
y2 = w2'*Z;

figure(7)
plot(1:44100,y2-son1_cr)
figure(13);
plot(1:44100,y2-son2_cr)

figure(8);
plot(1:44100,y2+son1_cr)
figure(9);
plot(1:44100,y2+son2_cr)

% cloud points and RSI computation
figure(9)
scatter(y1 ,son1_cr);

figure(10)
scatter(y2 , -son2_cr);

rsi = 5*log10(mean(son1_cr.^2,2)/mean((son1_cr-y1).^2,2)) + 5*log10(mean(son2_cr.^2,2)/mean((son2_cr+y2).^2,2));

% listen the solutions
sound(son1_cr)
pause(8)
sound(son2_cr)
pause(8)
sound(y1)
pause(8)
sound(y2)


%% Question 2: 
% 'Images separation'


im1 = double(imread('image_mix1.bmp'));
im2 = double(imread('image_mix2.bmp'));

im1_flat=reshape(im1,[1,1092*772]);
im2_flat=reshape(im2,[1,1092*772]);

%Centrage/Normalisation
im1_c = (im1_flat - mean(im1_flat)) ;
im1_cr = im1_c / sqrt(mean(im1_c.^2));
im2_c = (im2_flat - mean(im2_flat)) ;
im2_cr = im2_c / sqrt(mean(im2_c.^2));

X = [im1_cr; im2_cr];

% Bleaching
C = cov(X');
[V,D] = eig(C);
SQ_D = [1/sqrt(D(1,1)), 0; 0, 1/sqrt(D(2,2))];
M = SQ_D*V'; % because V is already an orthogonal matrix
Z = M*X;

epsilon = 0.000001;
p=1;

stock_w1 = zeros(2,n);
stock_y1 = zeros(n,843024);
stock_kurt_y1 = zeros(1,n);

w1_init = randn(2,1);

w1 = w1_init;
w1_next = mean(Z.*(w1'*Z).^3,2) - 3*w1;
w1_next = w1_next/sqrt(w1_next'*w1_next);
y1 = w1'*Z;
y1_next = w1_next'*Z;
kurt_y1 = kurtosis(y1);
kurt_y1_next = kurtosis(y1_next);

stock_w1(:,1:2) = [w1, w1_next];
stock_y1(1:2,:) = [y1; y1_next];
stock_kurt_y1(1,1:2) = [kurt_y1, kurt_y1_next];

while abs(stock_kurt_y1(p+1) - stock_kurt_y1(p)) > epsilon

    p = p+1;
    w1 = stock_w1(:,p);
    w1_next = mean(Z.*(w1'*Z).^3,2) - 3*w1;
    w1_next = w1_next/sqrt(w1_next'*w1_next);
    y1_next = w1_next'*Z;
    kurt_y1_next = kurtosis(y1_next);
    
    stock_w1(:,p+1) = w1_next;
    stock_y1(p+1,:) = y1_next;
    stock_kurt_y1(p+1) = kurt_y1_next;
    
end

p=p+1;

% plots of the sounds 1 and 2
x = 1:843024;
figure(20)
plot(x,im1_cr) 
figure(21)
plot(x,im2_cr-im1_cr) 

figure(22)
scatter(Z(1,:), Z(2,:));

% w1 components and kurt_y1
figure(1)
plot(1:n,stock_w1(1,:),1:n,stock_w1(2,:))

figure(2)
plot(1:n,stock_kurt_y1)

% Because of the indeterminations of permutations and signs, we plot here 4
% variables to emphasize the corresponding sources and estimations 
y1 = stock_y1(p,:);

figure(3)
plot(x,y1-im1_cr)
figure(4);
plot(x,y1-im2_cr) % good one to choose: y1 estimates im2_cr

figure(5);
plot(x,y1+im1_cr)
figure(6);
plot(x,y1+im2_cr)

% same process for y2
alpha = atan(-stock_w1(1,p)/stock_w1(2,p));
w2 = [cos(alpha); sin(alpha)];
y2 = w2'*Z;

figure(7)
plot(x,y2-im1_cr)
figure(13);
plot(x,y2-im2_cr) 

figure(8);
plot(x,y2+im1_cr) %% good one: y2 estimates -im1_cr
figure(9);
plot(x,y2+im2_cr)

% cloud points and RSI computation
figure(9)
scatter(y1 ,im2_cr);

figure(10)
scatter(y2 , -im1_cr);

rsi = 5*log10(mean(son1_cr.^2,2)/mean((im2_cr-y1).^2,2)) + 5*log10(mean(son2_cr.^2,2)/mean((im1_cr+y2).^2,2));

% plot the mixed images
image1 = imread('image_mix1.bmp');
image2 = imread('image_mix2.bmp');
figure;
imshow(image1);
figure;
imshow(image2);

% plot the estimations of the original images
im1_fin=reshape(-y2,[1092,772]);
im2_fin=reshape(y1,[1092,772]);
figure(13)
imshow(im1_fin*255)
figure(14)
imshow(im2_fin*255)


%% Question 3: 
% 'Artificial gaussian sources separation'

rand('seed', 1);

sig1 = randn(1,1000);
sig2 = randn(1,1000);

sig1_c = (sig1 - mean(sig1)) ;
sig1_cr = sig1_c / sqrt(mean(sig1_c .* sig1_c ));
sig2_c = (sig2 - mean(sig2)) ;
sig2_cr = sig2_c / sqrt(mean(sig2_c .* sig2_c ));

A = [1, 0.7; 0.8, 1];
S = [sig1_cr; sig2_cr];
X = A * S;

C = cov(X');
[V,D] = eig(C);
SQ_D = [1/sqrt(D(1,1)), 0; 0, 1/sqrt(D(2,2))];
M = SQ_D*V'; % because V is already an orthogonal matrix
Z = M*X;

rand('seed', 2);
z1 = Z(1,:);
z2 = Z(2,:);
p=1;
n = 50;

% criteria of convergence
epsilon = 0.000001;

stock_w1 = zeros(2,n);
stock_y1 = zeros(n,1000);
stock_kurt_y1 = zeros(1,n);

w1_init = randn(2,1);

% initialization of the two first w before launching the while structure
w1 = w1_init;
w1_next = mean(Z.*(w1'*Z).^3,2) - 3*w1;
w1_next = w1_next/sqrt(w1_next'*w1_next);
y1 = w1'*Z;
y1_next = w1_next'*Z;
kurt_y1 = kurtosis(y1);
kurt_y1_next = kurtosis(y1_next);

% initialization of the the stocking matrices
stock_w1(:,1:2) = [w1, w1_next];
stock_y1(1:2,:) = [y1; y1_next];
stock_kurt_y1(1,1:2) = [kurt_y1, kurt_y1_next];

while abs(stock_kurt_y1(p+1) - stock_kurt_y1(p)) > epsilon

    p = p+1;
    w1 = stock_w1(:,p);
    w1_next = mean(Z.*(w1'*Z).^3,2) - 3*w1;
    w1_next = w1_next/sqrt(w1_next'*w1_next);
    y1_next = w1_next'*Z;
    kurt_y1_next = kurtosis(y1_next);
    
    stock_w1(:,p+1) = w1_next;
    stock_y1(p+1,:) = y1_next;
    stock_kurt_y1(p+1) = kurt_y1_next;
    
end

p=p+1;

figure(6)
plot(1:n,stock_w1(1,:),1:n,stock_w1(2,:))

figure(7)
plot(1:n,stock_kurt_y1)

y1 = stock_y1(p,:);

% Because of the indeterminations of permutations and signs, we plot here 4
% variables to emphasize the corresponding sources and estimations
figure(8)
plot(1:1000,y1-sig1_cr)
figure(9);
plot(1:1000,y1-sig2_cr)

figure(10);
plot(1:1000,y1+sig1_cr)
figure(11);
plot(1:1000,y1+sig2_cr) % the good one: y1 estimates -sig2_cr

alpha = atan(-stock_w1(1,p)/stock_w1(2,p));
w2 = [cos(alpha); sin(alpha)];
y2 = w2'*Z;

figure(12)
plot(1:1000,y2-sig1_cr)
figure(13);
plot(1:1000,y2-sig2_cr)

figure(14);
plot(1:1000,y2+sig1_cr) % the good one: y2 estimates -sig1_cr
figure(15);
plot(1:1000,y2+sig2_cr)

figure(15)
scatter(y1 ,-sig2_cr);

figure(16)
scatter(y2 , -sig1_cr);

rsi = 5*log10(mean(sig1_cr.^2,2)/mean((sig1_cr+y2).^2,2)) + 5*log10(mean(sig2_cr.^2,2)/mean((sig2_cr+y1).^2,2));


% RSI = 3.43

% The algorithm is really less effective than in the original situation
% because of gaussian sources signal generation. Indeed, the SAS algorithms
% are actually designed for independant, identicly distributed and
% non-gaussian signals. So, it seems clear that new algorithms have to be developped to process gaussian signals. 
% To process gaussian sources, specific algorithms consist in supressing
% some of basic hypothesis.


%% VI. Separation of further dimensionned sources signals (N sources, N mixes) 






