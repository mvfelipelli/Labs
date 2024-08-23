
require(magrittr)



###############################################
### funcao base para alisamento exponencial ####
###############################################

## OBS: Ajuste via minimização da soma dos erros ao quadrado (EMQ)
####      Assim como nos modelos ARIMA, sob a suposição de normalidade 
####      o EMQ equivale ao  estimador de máxima verossimilhança (EMV)



#### Modelos de Holt - Winters 

## HW - Aditivo (default)
fit.a <- HoltWinters(x=AirPassengers, seasonal = "additive")
fit.a
plot(fit.a)			
			
## HW - Multiplicativo
fit.m <- HoltWinters(x=AirPassengers, seasonal = "multiplicative")
fit.m
plot(fit.m)			


### Previsão ###

## h=10
## Probabilidade cobertura = 95%
predict(fit.a, n.ahead = 10, prediction.interval = TRUE, level = 0.95)

predict(fit.m, n.ahead = 10, prediction.interval = TRUE, level = 0.95)



## OBS 3: Os modelos SES e Holt podem ser acessados da seguinte forma:

## SES
fit.ses <- HoltWinters(x=AirPassengers, beta=FALSE, gamma=FALSE)
fit.ses
plot(fit.ses)

## Holt
fit.holt <- HoltWinters(x=AirPassengers, gamma=FALSE)
fit.holt
plot(fit.holt)


## OBS 4: Tendencia Damped não está disponível na função base

## OBS 5: Plot das previsões não está diretamente disponível na função base


### Alternativa para plotar as previsões
x <- cbind(AirPassengers,predict(fit.ses,18),predict(fit.holt,18),predict(fit.a,18) )
colnames(x) <- c('AirPassengers','SES','Holt','HW-aditivo')

plot(x, plot.type='single', col=c(1,2,3,4), lwd=2)
legend(x=1950, y = 650, legend=colnames(x), box.lwd='white', col = c(1,2,3,4), text.col = c(1,2,3,4), lwd=2)


### HW-aditivo vs HW-multiplicativo
x <- cbind(AirPassengers,predict(fit.a,24),predict(fit.m,24) )
colnames(x) <- c('AirPassengers','HW-aditivo','HW-multiplicativo')

plot(x, plot.type='single', col=c(1,2,3), lwd=2, lty=c(1,2,3))
legend(x=1950, y = 700, legend=colnames(x), box.lwd='white', col = c(1,2,3), text.col = c(1,2,3), lwd=2, lty=c(1,2,3))








###############################################
###      Pacote forecast                   ####
###############################################

require(forecast)

## SES  --> ajuste e previsão
ses(y, h = 10, level = c(80, 95), initial = c("optimal", "simple"))

## Holt --> ajuste e previsão
holt(y, h = 10, damped = FALSE, level = c(80, 95), initial = c("optimal", "simple"))

## Holt+Damped --> ajuste e previsão
holt(y, h = 10, damped = TRUE, level = c(80, 95), initial = c("optimal", "simple"))

## Holt Winters --> ajuste e previsão
hw(y, h = 10, seasonal = c("additive", "multiplicative"),
  damped = FALSE, level = c(80, 95), initial = c("optimal", "simple"))
  
## Holt Winters + Damped --> ajuste e previsão
hw(y, h = 10, seasonal = c("additive", "multiplicative"),
  damped = TRUE, level = c(80, 95), initial = c("optimal", "simple"))
  
  
  
  
#############################
### Alguns exemplos       ###
#############################

require(tseries) ## para usar a função kpss.test()


#######################
# 1) AirPassengers   ##
#######################

## HW-aditivo
fit.hwa <- hw(AirPassengers, h=24, seasonal = "a", initial = "o")
summary(fit.hwa)
plot(fit.hwa)

## HW-multiplicativo
fit.hwm <- hw(AirPassengers, h=24, seasonal = "m", initial = "o")
summary(fit.hwm)
plot(fit.hwm)

## HW-multiplicativo + Damped
fit.hwmd <- hw(AirPassengers, h=24, seasonal = "m", initial = "o", damped=T)
summary(fit.hwmd)
plot(fit.hwmd)


### seleção do modelo via AICc (Critério de Akaike Corrigido)
fit.hwa$model$aicc
fit.hwm$model$aicc
fit.hwmd$model$aicc



### analise residual
E <- fit.hwm$residuals ## residuos do método selecionado

# visual
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
Box.test(E, lag = 15, type ="Ljung-Box", fitdf = 3) ## use fitdf=p+q
# normalidade
shapiro.test(E)






#####################################
# 2) airmiles        ################
#####################################

# Série anual de utilização de milhas aéries
plot(airmiles)

fit.ses <- ses(airmiles,h=5)
fit.ses$model
plot(fit.ses)

fit.holt <- holt(airmiles,h=5)
fit.holt$model
plot(fit.holt)

fit.damped <- holt(airmiles,h=5, damped=T)
fit.damped$model
plot(fit.damped)

### selecionando modelo utilizando AICc (Critério de Akaike Corrigido)

fit.ses$model$aicc
fit.holt$model$aicc
fit.damped$model$aicc

### analise residual
E <- fit.holt$residuals ## residuos do método selecionado

# visual
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
Box.test(E, lag = 15, type ="Ljung-Box", fitdf = 2) ## use fitdf=p+q
# normalidade
shapiro.test(E)




#####################################
# 3) USAccDeaths   ##################
#####################################

# serie mensal de mortes por acidentes nos EUA
plot(USAccDeaths)

## Visulamente, podemos notar:
# a) Série sazonal com mais mortes no meio do ano (verao) que no inicio (inverno)
# b) A amplitude da sazonalidade parece estabilizada 
# c) Não existe tendência de crescimento/queda

## Modelo indicado (visualmente) --> Holt Winters Aditivo
## Note que a componente de crescimento poderia ser descartada (beta=FALSE)

## HW-aditivo
fit.hwa <- hw(USAccDeaths, h=24, seasonal = "a")
fit.hwa$model

## HW-multiplicativo
fit.hwm <- hw(USAccDeaths, h=24, seasonal = "m")
fit.hwm$model

### selecionando modelo utilizando AICc (Critério de Akaike Corrigido)

fit.hwa$model$aicc

fit.hwm$model$aicc


### Plot modelo selecionado
plot(fit.hwa)


### analise residual
E <- fit.hwa$residuals ## residuos do método selecionado

# visual
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
Box.test(E, lag = 15, type ="Ljung-Box", fitdf = 3) ## use fitdf=p+q
# normalidade
shapiro.test(E)



