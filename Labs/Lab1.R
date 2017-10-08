#Part 1 - Bayesian linear regression

#Bayesian linear regression
# Model : Yi =⟨θ,φ(xi)⟩+εi/ (εi)i∈N ∼ N(0,β )

#Prior : θ ∼ N (0p, α−1Ip)
#L(Yi|θ) = N (⟨θ, φ(xi)⟩, β−1)
#Posterior : π( · |y1:n) = N (mn, Sn)
#mn, Sn are given in equation (3) lab1.pdf


glinear_fit <- function(Alpha, Beta, data, feature_map, target) 
  ## Alpha: prior precision on theta
  ## Beta: noise precision
  ## data: the input variables (x): a matrix with n rows
  #### where n is the sample size
  ##feature_map: the basis function, returning a vector of
  #### size p equal to the dimension of theta
  ## target: the observed values y: a vector of size n
{
  Phi <- t(mapply(feature_map, data))
  p = ncol(Phi)
  Ip = diag(p)
  posterior_variance <- solve(Alpha*Ip + Beta*t(Phi)%*%Phi)
  posterior_mean <- (solve(Alpha/Beta*Ip + t(Phi)%*%Phi)%*%t(Phi))%*%target
  return(list(mean=posterior_mean, cov=posterior_variance))
}

#Test glinear_fit

# Building Data 
#polynomial basis function model
Fmap <- function(x){c(1, x,x^2,x^3, x^4)}

Theta0 = c(5,2,1,-1,-0.1)
Beta0 = 1
Alpha0 = 1 # Precision on prior theta
n = 100

#Generate data set

generate_data <- function(n, Beta, Theta)
{
  eps <- matrix(rnorm(n , mean = 0, sd = sqrt(1/Beta0)))
  X <- matrix(runif(n,-3,3),nrow = n, ncol = 1) # Represents Data
  Phi <- t(mapply(Fmap,X))
  Y <- (Phi%*%Theta0 + eps) # Represents the target
  return(list(X=X, Y=Y))
}

data = generate_data(n, Beta0, Theta0)
Estimation <- glinear_fit(Alpha0, Beta0, data$X, Fmap, data$Y)


#Convergance
#Bahvior as a function of n

N = 1000
Fmap <- function(x){c(1, x,x^2,x^3)}

Theta0 = c(5,2,1,-1)
Beta0 = 1
Alpha0 = 1 # Precision on prior theta
data = generate_data(n = N, Beta = Beta0, Theta = Theta0)

step = 50
n = seq(from = 1,to = N, by= step)
m = length(n)

mean = 0
cov_diag = 0 
Interv = 0

for (i in 1:m){
  
  model_estimation <- glinear_fit(Alpha0, Beta0, data$X, Fmap, data$Y)
  local_data_X = head(data$X,n = i*step)
  local_data_Y = head(data$Y,n = i*step)
  estimate_sub_set <- glinear_fit(Alpha0, Beta0, local_data_X, Fmap, local_data_Y)
  mean = rbind(mean,estimate_sub_set$mean[,1])
  cov_diag = rbind(cov_diag , diag(estimate_sub_set$cov))
}

# Confidence Intervals

low = mean - 1.96*sqrt(cov_diag)
up =mean + 1.96*sqrt(cov_diag)


t = 4 # Indice of Thetha_t

xx <- 1:m
yy <- mean[,t][1:m+1] #Estimated Points 
pp <- rep(Theta0[t],m) #Real Point
lsup <- up[,t][1:m+1]
linf <-low[,t][1:m+1]
plot(xx, yy, lwd=1, ylim=range(lsup,linf))
lines(xx[order(xx)], yy[order(xx)], lwd=2)
lines(xx[order(xx)], lsup[order(xx)], col='red')
lines(xx[order(xx)], linf[order(xx)], col='red')
lines(xx, pp, pch=19,col='blue')
legend('bottom', legend=c('estimate', 'credible levels', 'Real Value'),
       col=c('black', 'red', 'blue'),
       pch= c(NA, NA, NA, 19),
       lwd=c(2,1,1,NA)
)



################ Part 2 - Predictive distribution ################
# L[Ynew |y1:n] = N (φ(xnew )'mn, φ(xnew )⊤'nφ(xnew ) +  1/B

glinear_pred <- function(Alpha, Beta, fitted, data, feature_map) ## Alpha: prior pecision for theta
  ## Beta: noise variance
  ## fitted: the output of glinear_fit: the posterior mean and
  #### variance of the parameter theta.
  ## data: new input data where the predictive distribution
  #### of Y must be computed
  ## feature map: the vector of basis functions
{
    Phi_transpose <- t(mapply(feature_map, data))
    pred_mean <- Phi_transpose %*% fitted$mean
    pred_variance <- Phi_transpose %*% fitted$cov %*% t(Phi_transpose) + 1/Beta
    return(list(mean = pred_mean, variance = diag(pred_variance)))
}


Fmap <- function(x){c(1, x,x^2,x^3, x^4)}

Theta0 = c(5,2,1,-1,-0.1)
Beta0 = 1
Alpha0 = 10 # Precision on prior theta
n = 100

#Generate data set and get the fitted model

data <- generate_data(n, Beta0, Theta0)
fitted_ <- glinear_fit(Alpha0, Beta0, data$X, Fmap, data$Y)

#Plot initial data and model
plot(data$X,data$Y)


#Prediction
nb = 50

X_new = matrix(seq(-3,3,length.out = nb))
Y_new = t(mapply(Fmap,X_new))%*%Theta0 + rnorm(nb, sd = sqrt(Beta0)^(-1))
nb_ = seq(from = 1,to = nb, by= 1)

Prediction = glinear_pred(Alpha = Alpha0, Beta = Beta0, fitted = fitted_, data = X_new, Fmap)

#Plot

low_pred = Prediction$mean - 1.96*sqrt(Prediction$variance)
up_pred =Prediction$mean + 1.96*sqrt(Prediction$variance)

plot(data$X,data$Y,xlab = 'X', ylab = 'Y')
lines(X_new,Prediction$mean,type = 'l',col='green')
lines(X_new, up_pred, col='red')
lines(X_new, low_pred, col='red')
lines(X_new,t(mapply(Fmap,X_new))%*%Theta0, col = 'purple')
points(X_new, Y_new, col='blue')
legend('bottomleft', legend=c('Initial data', 'credible levels 95%',  'Estimation for new set of points', 'New set of points', 'Exacte Model for new points'),
       col=c('black', 'red', 'green','blue','purple'),
       pch= c(NA, NA, NA, NA),
       lwd=c(1,1,1,1),
       ncol = 1,
       cex = 0.5
)

################ Part 3 - log-evidence and hyperparameters Alpha and Beta ################
## Empirical bayes / model selection: computation of the log-evidence. 

logevidence <- function(Alpha, Beta, data ,feature_map, target)
  ## Alpha: prior precision for theta
  ## Beta: noise precision
  ## data: the input points x_{1:n}
  ## feature_map: the vector of basis functions
  ## target: the observed values y: a vector of size n.  
{
  Phi_transpose <-  apply(X= data, MARGIN=1, FUN = feature_map)
  if(is.vector(Phi_transpose)){
    Phi_transpose = matrix(Phi_transpose,nrow=1)
  }
  Phi <- t(Phi_transpose)
  N <- nrow(Phi)
  p <- ncol(Phi)
  A <- Alpha*diag(p) + Beta * Phi_transpose %*% Phi
  postmean <- Beta * solve(A) %*% Phi_transpose %*% target
  energy <- Beta/2 * sum(( target - Phi%*%postmean)^2) + Alpha/2 * sum((postmean)^2)
  res <- p/2 * log(Alpha) + N/2 * log(Beta) - energy - 1/2 * log(det(A)) - N/2 * log(2*pi)
  return(res)    
}

logevidence2 <- function(Alpha, Beta, data ,feature_map, target)
{
  Phi_transpose <-  apply(X= data, MARGIN=1, FUN = feature_map)
  if(is.vector(Phi_transpose)){
    Phi_transpose = matrix(Phi_transpose,nrow=1)
  }
  target <- matrix(target, ncol=1)
  Phi <- t(Phi_transpose)
  N <- nrow(Phi)
  p <- ncol(Phi)
  Sigma <- Beta^{-1}*diag(N) + Alpha^{-1} * Phi %*% Phi_transpose
  ##    postmean <- Beta * solve(A) %*% Phi_transpose %*% target
  ##    energy <- Beta/2 * sum(( target - Phi%*%postmean)^2) + Alpha/2 * sum((postmean)^2)
  res <- -1/2 * (log(det(Sigma))  + N/2 * log(2*pi) + t(target)%*%solve(Sigma)%*%target )
  return(res)    
}


######## Empirical Bayes for Alpha ############

h0 <- function(x){ sum(Fmap(x)* theta0)}

ALPHA <- seq(0.5, 200,by=0.5)##exp(seq(-5,5,by=0.2))
Beta <- 5 ## 
N <- 20
##set.seed(3)
data <- matrix(runif(N, min=-3, max = 3),ncol=1)
Beta0 <- 5 ## true Beta
data <- generate_data(N, Beta0, Theta0)
X_ <- data$X
target <- data$Y
logevid_Alpha <- sapply(ALPHA,FUN=function(a){
  logevidence2(Alpha=a, Beta, X_ ,feature_map=Fmap, target)})
plot(ALPHA, logevid_Alpha)


BETA <- seq(0.1, 15,by=0.05)##exp(seq(-5,5,by=0.2))
Alpha <- 0.03 ## result  quite robust to a modification of alpha
N <- 20
##set.seed(3)
Beta0 <- 5 ## true Beta
data <- generate_data(N, Beta0, Theta0)
X_ <- data$X
target <- data$Y

logevid_Beta <- sapply(BETA,FUN=function(b){
  logevidence(Alpha, Beta=b, X_ ,feature_map=Fmap, target)})
plot(BETA, logevid_Beta)


#### Optimization on Alpha and Beta


N <- 200
##set.seed(3)5
data <- matrix(runif(N, min=-3, max = 3),ncol=1)
Beta0 <- 5 ## true Beta
##target <-  sin(data) + rnorm(N, sd = sqrt(Beta0)^(-1))
target <-  sapply(data,h0) + rnorm(N, sd = sqrt(Beta0)^(-1))
optAB <- optim(par=c(1, 1),
               fn=function(par){-logevidence(Alpha=par[1], Beta=par[2], data ,feature_map=Fmap, target)},
               method = "L-BFGS-B",
               lower=c(0.1 , 0.1), upper = c(250,30))
optAB



##### model choice: polynomial order.


F7<- function(x){c(1, x, x^2, x^3, x^4,x^5, x^6, x^7)}
F6 <- function(x){c(1, x, x^2, x^3, x^4,x^5, x^6)}
F5 <- function(x){c(1, x, x^2, x^3, x^4,x^5)}
F4 <- function(x){c(1, x, x^2, x^3, x^4)}
F3 <- function(x){c(1, x, x^2, x^3)}
F2 <- function(x){c(1, x, x^2)}
F1 <- function(x){c(1, x)}
F0 <- function(x){1}
listF=list(F0,F1,F2,F3,F4,F5,F6,F7)

data <- matrix(runif(N, min=-3, max = 3),ncol=1)
Beta0 <- 5 ## true Beta
#target <-  sin(data) + rnorm(N, sd = sqrt(Beta0)^(-1))
target <-  sapply(data,h0) + rnorm(N, sd = sqrt(Beta0)^(-1))
logevid_p <- sapply(0:7,FUN=function(i){
  logevidence(Alpha=10, Beta=Beta0, data ,feature_map=listF[[i+1]], target)})
plot(0:7, logevid_p)

which.max(logevid_p)


#### Model choice joint Optimization

N <- 200
data <- matrix(runif(N, min=-3, max = 3),ncol=1)
Beta0 <- 5 ## true Beta
target <-  sapply(data,h0) + rnorm(N, sd = sqrt(Beta0)^(-1))
##target <-  sin(data) + rnorm(N, sd = sqrt(Beta0)^(-1))
Alphastars <- rep(0,8)
Betastars <- rep(0,8)
values <- rep(0,8)
for(i in 1:8){
  optAB <- optim(par=c(1, 1),
                 fn=function(par){-logevidence(Alpha=par[1], Beta=par[2], data ,feature_map=listF[[i]], target)},
                 method = "L-BFGS-B",
                 lower=c(0.1 , 0.1), upper = c(300,50))
  Alphastars[i] <- optAB$par[1]
  Betastars[i] <- optAB$par[2]
  values[i] <- - optAB$value
}

plot(values)
