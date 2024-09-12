
####################################################################
########## Simulando e Ajustando um processo ARIMA(p,d,q) ##########
####################################################################


# Um ARIMA(p,d,q) pode ser simulado em R utilizando a função 
# 	arima.sim(n, model)
# Por exemplo o comando
x <- arima.sim(n = 1000, model=list(order=c(1,1,1), ar = 0.6, ma = 0.5))
# simula o modelo ARMA(1,1,1)
# W_t = X_t - X_{t-1}
# W_t = 0.6 W_{t-1} + E_t + 0.5 E_{t-1},   E_t ~ N(0,1) 


# Para definir um valor diferente para sigma2, por exemplo, sigma2=5, utilize
y <- arima.sim(n = 1000, model=list(ar = 0.6, ma = 0.5), 
		rand.gen = function(n) rnorm(n,0,sqrt(5)) )


par(mfrow=c(2,1))
plot(x)
plot(y)



# Da mesma forma, a função arima(order=c(p,d,q)) pode ser 
# utilizada para ajustar o modelo ARIMA(p,d,q) via minimos quadrados
# Por exemplo,
arima(x, order = c(1, 1, 1), include.mean = FALSE)

arima(y, order = c(1, 0, 1), include.mean = FALSE)




### A função ndiffs() do pacote 'forecast' mede quantas vezes a série 
## precisa ser diferenciada
## até se tornar estacionária de acordo com um dos testes estatísticos
## kpss, adf e pp
require(forecast)

## ARMA(1,1)
x <- arima.sim(n = 1000, model=list(order=c(1,0,1), ar = 0.6, ma = 0.5))

## ARIMA(1,1,1)
y <- arima.sim(n = 1000, model=list(order=c(1,1,1), ar = 0.6, ma = 0.5))

## ARIMA(1,2,1)
z <- arima.sim(n = 1000, model=list(order=c(1,2,1), ar = 0.6, ma = 0.5))

## testes
ndiffs(x, alpha = 0.05, test = c("kpss", "adf", "pp"), max.d = 2)
ndiffs(y, alpha = 0.05, test = c("kpss", "adf", "pp"), max.d = 2)
ndiffs(z, alpha = 0.05, test = c("kpss", "adf", "pp"), max.d = 2)






####################################################################
#########  Redundância dos parâmetros ##############################
####################################################################

### Exemplo: Shumway book
set.seed(8675309)
x = rnorm(150, mean=5) # generate iid N(5,1)s

### Era de se esperar que a estimativa fosse ar1 = 0 e ma1 =0, no entanto
### a função de estimação se perde e encontra
### ar_1 = -0.9595 e ma_1 = 0.9527
arima(x, order=c(1,0,1)) # estimation ARMA(1,1)

### Note que a proximidade do termo ar_1 e ma_1
### indica que o modelo está super parametrizado, 
### pois os fatores quase podem ser cancelados na forma polinomial
### (1 + 0.96B) X_t = (1+0.95B) E_t

### Observe também o tamanho do erro padrão do estimador


### Nesta situação, os gráficos de correlações mostram que
### não existe correlações para nenhuma defasagem
ndiffs(x) 
par(mfrow=c(3,1))
plot.ts(x)
print( acf(x))
print( pacf(x))




