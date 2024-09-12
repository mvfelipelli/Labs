
####################################################################
#########   Adequação de modelos ARIMA(p,d,q)          #############
####################################################################

## simulando ARIMA(1,1,1)
x <- arima.sim(n = 1000, model=list(order=c(1,1,1), ar = 0.6, ma = 0.5))

## testando número de diferenças necessárias
require(forecast)
d <- ndiffs(x, alpha = 0.05)

## série diferenciada
w <- diff(x, differences = d )

## correlações da série estacionária {w_t}
par(mfrow=c(3,1))
plot.ts(w)
print( acf(w))
print( pacf(w))


require(tseries) ## para usar a função kpss.test()


#################################################
### verificando um modelo AR(p) para {w_t}  #####
#################################################
fit.ar <- arima(w, order=c(4,0,0), include.mean = FALSE)

E <- fit.ar$residuals

## análise visual
windows()
par(mfrow=c(2,2))
plot(E)
acf(E)
pacf(E)
qqnorm(E)
abline(0,1)

## testes estatisticos
# Estacionaridade
kpss.test(E) # hipótese nula: série estacionária
# independencia
Box.test(E, lag = 20, type ="Ljung-Box", fitdf = 4) ## use fitdf=p+q
# normalidade
shapiro.test(E)



#################################################
#### verificando um modelo MA(q) para {w_t} #####
#################################################
fit.ma <- arima(w, order=c(0,0,6), include.mean = FALSE)

E <- fit.ma$residuals

## análise visual
windows()
par(mfrow=c(2,2))
plot(E)
acf(E)
pacf(E)
qqnorm(E)
abline(0,1)

## testes estatisticos
# Estacionaridade
kpss.test(E) # hipótese nula: série estacionária
# independencia
Box.test(E, lag = 20, type ="Ljung-Box", fitdf = 6) ## use fitdf=p+q
# normalidade
shapiro.test(E)



#################################################
### verificando um modelo ARMA(1,1) para {w_t} ##
#################################################
fit.arma <- arima(w, order=c(1,0,1), include.mean = FALSE)

E <- fit.arma$residuals

## análise visual
windows()
par(mfrow=c(2,2))
plot(E)
acf(E)
pacf(E)
qqnorm(E)
abline(0,1)

## testes estatisticos
# Estacionaridade
kpss.test(E) # hipótese nula: série estacionária
# independencia
Box.test(E, lag = 20, type ="Ljung-Box", fitdf = 2) ## use fitdf=p+q
# normalidade
shapiro.test(E)




##################################################
### Comparando os modelos                      ###
##################################################
fit.ar
fit.ma
fit.arma


