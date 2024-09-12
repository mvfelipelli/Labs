
####################################################################
##########     Modelos SARIMA(p,d,q)x(P,D,Q)           #############
####################################################################

x <- AirPassengers
plot(x)

####################
#### diferenças ####
####################
require(tseries) ## para usar a função kpss.test()

# primeira diferença
dx <- diff(x)

par(mfrow=c(3,1))
plot(dx, main='dx')
acf(dx, lag=5*12)
pacf(dx, lag=5*12)

kpss.test(dx)    # hipótese nula: série estacionária



# diferença sazonal
require(forecast)

nsdiffs(dx) ## ou nsdiffs(x) --> tendencia removida por decomposicao

d12dx <- diff(dx,lag=12)
par(mfrow=c(3,1))
plot(d12dx, main='d12dx')
acf(d12dx, lag=5*12)
pacf(d12dx, lag=5*12)




################################
### Modelos Candidatos #########
################################

# Sabemos que d=1 e D=1
#
# demais argumentos:
# p = 1      --> ACF decaindo rapidamente para zero e PACF com pequena "quebra" no lag 1
# q = 0 ou 1 --> PACF com "quebra" no lag 1
# P = 0      --> ACF e PACF sem autocorelações nos lags sazonais
# Q = 0      --> ACF e PACF sem autocorelações nos lags sazonais  


## Modelos candidatos para {x}
# --> SARIMA (1,1,0) x (0,1,0)
# --> SARIMA (1,1,1) x (0,1,0)


##############################################
## Modelo 1: SARIMA (1,1,0) x (0,1,0)
##############################################
fit1 <- arima(x, order=c(1,1,0), seasonal = c(0,1,0), include.mean = FALSE)
fit1

# resíduos
E <- fit1$residuals
plot(E)

E <- window(E,  start=time(x)[14])
plot(E)


## análise visual
par(mfrow=c(2,2))
plot(E)
acf(E)
pacf(E)
qqnorm(E)
qqline(E)

## testes estatisticos
# Estacionaridade
kpss.test(E) # hipótese nula: série estacionária
# independencia
Box.test(E, lag = 15, type ="Ljung-Box", fitdf = 1) ## use fitdf=p+q
# normalidade
shapiro.test(E)





##############################################
## Modelo 2: SARIMA (1,1,1) x (0,1,0)
##############################################
fit2 <- arima(x, order=c(1,1,1), seasonal = c(0,1,0), include.mean = FALSE)
fit2

# resíduos
E <- fit2$residuals
plot(E)

E <- window(E,  start=time(x)[14])
plot(E)


## análise visual
par(mfrow=c(2,2))
plot(E)
acf(E)
pacf(E)
qqnorm(E)
qqline(E)

## testes estatisticos
# Estacionaridade
kpss.test(E) # hipótese nula: série estacionária
# independencia
Box.test(E, lag = 15, type ="Ljung-Box", fitdf = 1) ## use fitdf=p+q
# normalidade
shapiro.test(E)




###########################################
### Criterio de seleção de modelos: AIC ###
###########################################
fit1 # --> escolhido pelo AIC
fit2






##########################################################################
####### Exercício: Verifique o melhor modelo para log(AirPassengers)  ####
##########################################################################


