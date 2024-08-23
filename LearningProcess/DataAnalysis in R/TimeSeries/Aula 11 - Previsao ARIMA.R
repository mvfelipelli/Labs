
## simulando ARIMA(1,1,1)
x <- arima.sim(n = 100, model=list(order=c(1,1,1), ar = 0.6, ma = 0.5))


## série diferenciada
w <- diff(x, differences = 1 )

## correlações da série estacionária {w_t}
par(mfrow=c(3,1))
plot.ts(w)
print( acf(w))
print( pacf(w))




#################################################
#### Previsão  do modelo MA(q) para {w_t} #######
#################################################
fit_w <- arima(w, order=c(0,0,6), include.mean = FALSE)

E <- fit_w$residuals

## análise visual
par(mfrow=c(2,2))
plot(E)
acf(E)
pacf(E)
qqnorm(E)
abline(0,1)


# Modelo escolhido para X --> ARIMA(0,1,6)
fit_x <- arima(x, order=c(0,1,6), include.mean = FALSE)

##################
### Previsão #####
##################
par(mfrow=c(2,1))

## Funcao base do R para previsão
## previsão série {w}
prev_w <- predict(fit_w, 10) ## previsão de 10 valores a frente de w_n

plot(cbind(w,prev_w$pred) , plot.type = "single", col=c(1,2), ylab='W')

## previsão série {x}
prev_x <- predict(fit_x, 10) ## previsão de 10 valores a frente de x_n

plot(cbind(x,prev_x$pred) , plot.type = "single", col=c(1,2), ylab='X')







######################################################
#### Ajuste e resíduos do ARMA(p,q) para {w_t}   #####
######################################################

fit_w <- arima(w, order=c(1,0,1), include.mean = FALSE)

E <- fit_w$residuals

## análise visual
par(mfrow=c(2,2))
plot(E)
acf(E)
pacf(E)
qqnorm(E)
abline(0,1)


# Modelo escolhido para X --> ARIMA(1,1,1)
fit_x <- arima(x, order=c(1,1,1), include.mean = FALSE)


###############################################
### Previsão da série {w_t} --> ARMA(1,1) #####
###############################################

par(mfrow=c(2,1))

## Funcao base do R para previsão

## previsão série {w}
prev_w <- predict(fit_w, 10) ## previsão de 10 valores a frente de w_n

plot(cbind(w,prev_w$pred) , plot.type = "single", col=c(1,2), ylab='W')

#### Calculando a variancia do ARMA(p,q) via MA(Inf)
sigma2 <- fit_w$sigma2 ## estimativa do sigma2
MA_coef <- ARMAtoMA(ar = 0.5571, ma = 0.5419, lag.max=9) ## tranforma ARMA(p,q) em MA(Inf) e retornar os coeficientes
var_cond <- sigma2*cumsum(c(1,MA_coef^2))
print(var_cond) #sigma2_{n+h|n}, h=1,2,...,10

## a função predict() retorna desvio padrão condicional (erro padrão), veja a equivalencia:
sqrt(var_cond)
prev_w$se


##################################################
### Previsão da série {x_t} --> ARIMA(p,d,q) #####
##################################################

# sabemos que x_{n+h|n} = x_n + soma_{i=1}^h w_{n+i|n}
# logo
n <- length(x)
x[n] + cumsum(prev_w$pred)
# são as previsões pontuais para a série {x}

## compare com os resultados da funcao predict()
prev_x <- predict(fit_x, 10) ## previsão de 10 valores a frente de x_n
prev_x$pred

plot(cbind(x,prev_x$pred) , plot.type = "single", col=c(1,2), ylab='X')

## a função predict retorna desvio padrão condicional (erro padrão) do modelo ARIMA(p,d,q)
## calculado via filtro kalman
prev_x$se



################################################################
###        Previsao intervalar                ##################
################################################################
###
### Verificada a condição de normalidade dos resíduos
### intervalos com 95% de confiança para X_{n+h|n}
### são dados por x_{n+h|n} +- 1,96 sqrt( sigma2_{n+h|n} )

par(mfrow=c(2,1))

### série {w}
LI <- prev_w$pred - 1.96*prev_w$se
LS <- prev_w$pred + 1.96*prev_w$se
plot(cbind(w,prev_w$pred,LI,LS) , plot.type = "single", col=c(1,2,3,3), ylab='W')

### série {x}
LI <- prev_x$pred - 1.96*prev_x$se
LS <- prev_x$pred + 1.96*prev_x$se
plot(cbind(x,prev_x$pred,LI,LS) , plot.type = "single", col=c(1,2,3,3), ylab='X')










################################################################
##########   O pacote forecast                ##################
################################################################

require(forecast)

par(mfrow=c(2,1))

### série {w}
prev_w <- forecast(fit_w, h=10, level=c(80,95))
prev_w
plot(prev_w, ylab='W')

### série {x}
prev_x <- forecast(fit_x, h=10, level=c(80,95))
prev_x
plot(prev_x, ylab='X')


#########################################
######## arima automático   #############
#########################################
## ajusta automaticamente as ordens e ajusta o modelo
fit_auto <- auto.arima(x)
E <- fit_w$residuals

## análise visual
par(mfrow=c(2,2))
plot(E)
acf(E)
pacf(E)
qqnorm(E)
abline(0,1)

prev_auto <- forecast(fit_auto, h=10, level=c(80,95))
prev_auto
plot(prev_auto)






#################################################################
######### AirPassengers #########################################
#################################################################

fit_auto <- auto.arima(AirPassengers)
E <- fit_w$residuals

## análise visual
par(mfrow=c(2,2))
plot(E)
acf(E)
pacf(E)
qqnorm(E)
abline(0,1)

prev_auto <- forecast(fit_auto, h=10, level=c(80,95))
prev_auto
plot(prev_auto)







#################################################################
#########      Nile          ####################################
#################################################################

plot(Nile)

fit_auto <- auto.arima(Nile)
E <- fit_w$residuals

## análise visual
par(mfrow=c(2,2))
plot(E)
acf(E)
pacf(E)
qqnorm(E)
abline(0,1)

prev_auto <- forecast(fit_auto, h=10, level=c(80,95))
prev_auto
plot(prev_auto)




require(magrittr)

x <- AirPassengers
fit1 <- arima(x, order=c(1,1,0), seasonal = c(0,1,0), include.mean = FALSE)
fit1

require(forecast) 

fit1 %>% forecast(36) %>% plot()


