###############################
#####  exemplos ETS  ##########
###############################

require(forecast)


#### AirPassengers ###

## ajuste
fit <- ets(AirPassengers)

 
summary(fit)

## plot das componentes
plot(fit)




### analise de residuos
E <- fit$residuals ## residuos do método selecionado

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
Box.test(E, lag = 15, type ="Ljung-Box", fitdf = 3)
# normalidade
shapiro.test(E)



par(mfrow=c(1,1))


### previsão de dois anos (h=24 meses)
prev <- forecast(fit, h=24, level = c(80, 95))

summary(prev)

plot(prev)


### previsões intervalares não paramétricas (bootstrap via reamostragem dos erros)
prev <- forecast(fit, h=24, level = c(80, 95), bootstrap = TRUE)

summary(prev)

plot(prev)




#### outras séries
# install.packages('fpp2')
require(fpp2)

plot(bicoal)

plot(chicken)

plot(dole)

plot(usdeaths)

plot(bricksq)

plot(lynx)

plot(eggs)

plot(ausbeer)

plot(debitcards)



