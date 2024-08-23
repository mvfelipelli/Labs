#Series temporais e analises preditivas - Fernando Amaral

library(forecast)
library(ggplot2)


#gerando serie random walk
#tornar reprodutivel
set.seed(4312)
x = cumsum(sample(c(-1,1),100,T))
print(x)
serie = ts(x,start = c(1901),end = c(2000), frequency = 1)
print(serie)
autoplot(serie)


#faz a previsao, h he o intervalo afrente a ser previsto
prev = naive(serie, h=5)
class(prev)
#observar que a previs?o pontual ? o ultimo valor da serie
print(prev)
print(prev$fitted)
print(prev$residuals)
autoplot(prev)
print(prev)

80  85
#altera o intervalo de confianca
prev2 = naive(serie, h=5, level = c(95, 99))
#observar que previsao pontual nao mudou, apenas os intervalos
print(prev2)
#comparando os plots
autoplot(prev2)

#comparando os dois plots
split.screen( figs = c( 2, 1 ) )
screen(1)
plot(prev)
screen(2)
plot(prev2)
close.screen( all = TRUE )

#naive sazonal
autoplot(AirPassengers)
prev3 = snaive(AirPassengers, h=12)
print(prev3)
autoplot(prev3)

#comparando a previs?o com o ultimo ano
prev3$mean
window(AirPassengers,start=c(1960))




