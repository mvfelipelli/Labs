

######## additive ####
X <- AirPassengers

dec <- decompose(X, type="a")

plot(dec)




######## multiplicative ####
X <- AirPassengers

dec <- decompose(X, type="multiplicative")

plot(dec)




######## Tendência: Linear ####

X <- AirPassengers
tempo <- time(X)

out <- lm(X ~ tempo)

plot(X)
lines(x=tempo, y=out$fitted, type='l', col='red', lwd=2, lty=2)



######## Tendência: Polinomial ####

X <- Nile
tempo <- as.numeric( time(X) )

out <- lm(X ~ poly(tempo,2))
out2 <- lm(X ~ poly(tempo,6))

plot(X, main='Flow of the River Nile')
lines(x=tempo, y=out$fitted, type='l', col='red', lwd=2, lty=2)
lines(x=tempo, y=out2$fitted, type='l', col='blue', lwd=2, lty=2)

legend(x=1950, y = 1300, legend = c('q=2','q=6'), col=c('red','blue'), lwd=2,  bty = "n")


####### Media Móvel Simples ###
X <- AirPassengers
sma6 <- filter(X, rep(1/6, 6), sides=1)

sma12 <- filter(X, rep(1/12, 12),sides=1)

plot(cbind(X,sma6,sma12), main='Air Passengers',plot.type='s', col=c('black','red','blue'), lwd=2, lty=c(1,2,1),ylab='')
legend(x=1950, y = 550, legend = c('SMA(6)','SMA(12)'), col=c('red','blue'), lwd=2,  bty = "n", lty=c(2,1))




######## Simétrico vs Passados #######

X <- AirPassengers
pass5 <- filter(X, rep(1/5, 5), sides=1)

sim5 <- filter(X, rep(1/5, 5), sides=2)

plot(cbind(X,pass5,sim5), main='Air Passengers',plot.type='s', col=c('black','red','blue'), lwd=2, lty=c(1,1,1),ylab='')
legend(x=1950, y = 550, legend = c('MM(5): valores passados','MM(5): Simétrico'), col=c('red','blue'), lwd=2,  bty = "n", lty=c(1,1))



######## Tendência Ajustada #######
X <- AirPassengers
T <- filter(X, rep(1/13, 13), sides=2)
Z <- X - T

plot(cbind(X,T,Z), main='Air Passengers',plot.type='m', lwd=2, lty=c(1,1,1),ylab='')



### Médias Sazonais de Z
n_periodos <- length(X)/12
MS <- t(matrix(data = Z, nrow = 12))
S <- colMeans(MS, na.rm = TRUE)
S <- ts(rep(S,n_periodos), start=start(X), frequency=frequency(X))

E <- X - T - S

plot(cbind(X,T,S,E), main='',plot.type='m', lwd=2)






####################################################
########## Exercício 1 #############################
####################################################

##### Decomposição Multiplicativa ####

X <- AirPassengers

T <- filter(X, rep(1/13, 13) , sides=2)

Z <- X/T


n_periodos <- length(X)/12

MS <- t(matrix(data = Z, nrow = 12))

S <- colMeans(MS, na.rm = TRUE)

S <- ts(rep(S,n_periodos), start=start(X), frequency=frequency(X))


plot(S)

E <- X/(T*S)

plot(cbind(X,T,S,E))





##### Decomposição Multiplicativa: Tendencia Polinomial ####

X <- Nile

tempo <- as.numeric( time(X) )

out <- lm(X ~ poly(tempo,5))

T <- out$fitted.values

T <- ts(T, start=start(X), frequency=frequency(X))

plot(cbind(X,T), plot.type='s', col=c(1,2))


Z <- X/T


n_periodos <- length(X)/12

MS <- t(matrix(data = Z, nrow = 12))

S <- colMeans(MS, na.rm = TRUE)

S <- ts(rep(S,n_periodos), start=start(X), frequency=frequency(X))


plot(S)

E <- X/(T*S)

plot(cbind(X,T,S,E))














