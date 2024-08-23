
require(magrittr)
require(quantmod)

### séries
getSymbols(c("PETR4.SA","VALE3.SA","BBAS3.SA"), src='yahoo', from=as.Date('2010-01-01'))
x <- cbind( Cl( PETR4.SA ), Cl( VALE3.SA ) , Cl( BBAS3.SA ) )
names(x) <- c('PETR4','VALE3','BBAS3')
x %>% tail()

### retornos
y <- diff( log(x) ) %>% na.omit()
y %>% head()
y %>% tail()  


#### H_t
Ht <- cov( y )

### pesos w's
custodia <- c(500, 300, 250)
capital_por_acao <-  tail(x,1)*custodia
capital <- sum(capital_por_acao)
w <- capital_por_acao / capital 

## ht - Volatilidade da carteira
ht <- as.numeric(w) %*% Ht %*% as.numeric(w)

## VaR 95% da carteira
VaR_95_perc <- -qnorm(0.05, mean=0, sqrt(ht)) # percentual
VaR_95_perc

VaR_95_capit <- VaR_95_perc*capital
VaR_95_capit



############ Exercício 2 ###########################
#install.packages('bayesDccGarch')
require('bayesDccGarch')

fit <- bayesDccGarch(10*y, nSim=10000)


fit2 = increaseSim(fit, 50000)


fit3 = window(fit2, start=10000, thin=5)

#plot(fit3$MC)

summary(fit3)

plot(fit3)

### ultimo dia ####
Ht <- fit3$H_n1 / 100

ht <- as.numeric(w) %*% Ht %*% as.numeric(w)

VaR_95 <- -qnorm(0.05, mean=0, sd=sqrt(ht))*capital


### VaR Historico ###

###### valor da Carteira ##################
C <- xts(x %*% custodia, order.by = index(x)) ## capital investido
head(C)

plot(C,main='Valor da carteira')


###### pesos #############################
w <- matrix(NA, nrow=nrow(x), ncol=ncol(x))
for(i in 1:nrow(x))
  w[i,] <- (custodia*x[i,])/as.numeric(C[i])

colnames(w) <- colnames(x)
head(w)  
tail(w)

#### volatilidade da carteira ###########
ht <- numeric(nrow(fit3$H))
w <- tail(w,nrow(fit3$H))
for(i in 1:nrow(fit3$H)){
  Ht <- matrix( fit3$H[i,], 3, 3) / 100
  
  ht[i] <- w[i,] %*% Ht %*% w[i,]
}

tail(ht)

#### Histórico do VaR 95% da carteira 
VaR95 <-  qnorm(0.05, mean=0, sd=sqrt(ht)) 
VaR95 <- xts(VaR95, order.by = index(y))
head(VaR95)
tail(VaR95)

#### Retornos da carteira ####
yy <- rowSums( w*y )
yy <- xts(yy, order.by = index(y))
head(yy)

plot(yy)

### Gráficos
windows()
par(mfrow=c(2,1))
plot( window(C,start=as.Date('2013-01-01')), main='Valor da carteira (R$)' )
plot( window(cbind(C*yy, C*VaR95),start=as.Date('2013-01-01')),  main='Retornos Financeiros e VaR(95%) (R$)', 
      multi.panel = FALSE)

### proporção de violações
mean( na.omit(yy < VaR95)["2013::"] )

