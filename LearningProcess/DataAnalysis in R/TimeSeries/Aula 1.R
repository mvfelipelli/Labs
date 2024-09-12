
set.seed(5)


############################################################
########     ATIVIDADE 1   #################################
############################################################

n <- 300  # tamanho da série


##### Item 1 ####################

X <- 1000 + 0.1*(1:n)

plot.ts(X)



##### Item 2 ####################

# série ciclica
S <- 20*sin((2*pi/12)*(1:n))
plot.ts(S)

# adicionando ao processo do Item 1
plot.ts(X+S)


##### Item 3 ####################
# ruido branco
E <- rnorm(n,0,5)
plot.ts(E)


# adicionando ao processo do Item 2
Z <- X+S+E
plot.ts(Z)


##### Item 4 ####################

# Série Original
Z <- ts(Z, start=1990, frequency=12)
plot(Z)

plot(decompose(Z))



##### Item 5 ####################

## Ideia de previsão:
# 1) extrapolar um modelo linear para a componente de tendência
# 2) repetir a componente de sazonalidade
# 3) usar zero como previsão da série de ruídos
# 4) somar as previsões das componentes para ter a previsão da série Z

h <- 30

dec <- decompose(Z)

trend <- dec$trend  ## componente de tendencia
sea <- dec$seasonal ## componente sazonal
E <- dec$random     ## componente aleatorio


## previsao da componente de tendencia
tempo <- 1:n
trend.lm <- lm(trend ~ tempo) ## linear model (lm)
trend.forec <- trend.lm$coefficients[[1]] +  trend.lm$coefficients[[2]]*((n+1):(n+h)) %>% ts(start=end(trend)+c(0,1), frequency = 12)
c(trend, trend.forec) %>% plot.ts()

## previsao do termo sazonal
library(forecast)

sea.forec <- snaive(sea, h=h)$mean    
# forecast - previsão

cbind(sea, sea.forec) %>% plot.ts(plot.type='s', col=c(1,2))


### previsão pontual
forec <- trend.forec + sea.forec

cbind(Z, forec) %>% plot.ts(plot.type = 's', col=c(1,2))







############################################################
########     ATIVIDADE 2   #################################
############################################################

### Item 1 ####
n <- 100

X <- numeric(100)

X[1:6] <- c(58.9,  60.0,  59.8,  59.2,  58.9,  58.7)

for(i in 7:n){
	X[i] <- 5 + 0.35*X[i-1] + 0.6*X[i-6] + rnorm(1,0,2)
}

X <- ts(X,start=2000,frequency=6)

plot(X)





###############################################
#############  Item 2 #########################
###############################################
X.obs <- window(X,end=time(X)[90])
X.teste <- window(X,start=time(X)[91])

plot(X.obs)
plot(X.teste)


## Bootstrap
n_sim = 200 # numero de simulacoes
X_matriz <- matrix(NA,nrow=n,ncol=n_sim) # matriz de possibilidades

# Preenche valores observados
for(j in 1:n_sim)
	X_matriz[1:90,j] <- X.obs 

# Simula dados futuros
for(i in 91:100){
	X_matriz[i,] <- 5 + 0.35*X_matriz[i-1,] + 0.6*X_matriz[i-6,] + rnorm(n_sim,0,2)
}	

# Plota todas as simulações
X_matriz <- ts(X_matriz,start=2000,frequency=6)

plot(X_matriz, plot.type='single')


## Banda com 95% de probabilidade
LI <- LS <- Pontual <- numeric(10)  
for(i in 1:10){
	LI[i] <- quantile(X_matriz[90+i,], prob=0.025)
	LS[i] <- quantile(X_matriz[90+i,], prob=0.975)
	Pontual[i] <- mean(X_matriz[90+i,])
}

prev_mat <- cbind(LI,Pontual,LS)
colnames(prev_mat) <- c('LI95','Pontual','LS95')

prev_mat <- ts(prev_mat, start=time(X)[91], frequency=6)


plot(X)

points(prev_mat[,1], col=3, type='l')
points(prev_mat[,2], col=2, type='l')
points(prev_mat[,3], col=3, type='l')


