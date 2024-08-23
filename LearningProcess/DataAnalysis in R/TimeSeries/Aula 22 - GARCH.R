
#install.packages('quantmod','fGarch')
require(magrittr)
require(quantmod)
require(fGarch)


### Retornos diários (exemplo: PETR4)
fonte <- 'yahoo'                  ## fonte dos dados: banco de dados do yahoo
inicial <- as.Date('2008-01-01')  ## data inicial
series <- new.env()

## Indice Ibovespa --> Indice das principais ações negociadas na bolsa Bovespa
getSymbols("PETR4.SA", env=series, src=fonte, from=inicial)

## Calculo dos retornos
y <- diff(log(Cl(series$PETR4.SA)))
y <- na.omit(y)

plot(y)

acf(y)

acf(y^2)


### Teste de Ljung-Box sobre {y_t^2}
# H_0: {y_t} é homocedástico
# H_1: {y_t} é heterocedástico
Box.test(y^2, lag = 15, type ="Ljung-Box", fitdf = 0)





########### GARCH(1,1) com erros normais #####################

fit <- garchFit(formula = ~garch(1,1), data=y, include.mean = FALSE, cond.dist='norm')

summary(fit)


### resíduos padronizados
stdResiduals <- fit@residuals/sqrt(fit@h.t)

plot.ts(stdResiduals)

acf(stdResiduals^2)

pacf(stdResiduals^2)

### aderência ###
qqnorm(stdResiduals)
qqline(stdResiduals)

shapiro.test(stdResiduals)

ks.test(stdResiduals, 'pnorm')

hist(stdResiduals, freq = FALSE)
curve(dnorm(x), add=TRUE)


### Gráfico de Retornos + Volatilidade
plot( window(cbind(y, 10*fit@h.t),start=as.Date('2017-01-01')), main='Retornos e 100xVolatilidade', 
      multi.panel = TRUE)


### Gráfico de Retornos + VaR(95%)
VaR_95 <- qnorm(0.05,mean=0,sd=sqrt(fit@h.t))
VaR_95 <- xts(VaR_95, order.by=index(y))

plot( window(cbind(y, VaR_95),start=as.Date('2017-01-01')), main='Retornos e VaR(95%)')

# taxa de cobertura
mean(y < VaR_95)


### Previsão da volatilidade
forec = predict(fit, n.ahead=252)
forec$standardDeviation^2 %>% plot()
#############################################################





###### GARCH(1,1) com erros t-student assimetrica padronizada (sstd) #########
fit <- garchFit(formula = ~garch(1,1) ,data=y, include.mean = FALSE, cond.dist='sstd')

summary(fit)

EMV <- fit@fit$par


### resíduos padronizados
stdResiduals <- fit@residuals/sqrt(fit@h.t)

plot.ts(stdResiduals)

acf(stdResiduals^2)

mean(stdResiduals)
sd(stdResiduals)

### aderência ###
x <- rsstd(n=1000, mean=0,sd=1,nu=EMV["shape"], xi=EMV["skew"])
qqplot( stdResiduals,  x , main='QQPlot SSTD')
qqline(stdResiduals)

ks.test(stdResiduals, 'psstd', mean=0,sd=1,nu=EMV["shape"], xi=EMV["skew"] )

hist( stdResiduals, freq = FALSE )
curve(dsstd( x, mean=0,sd=1,nu=EMV["shape"], xi=EMV["skew"]), add=TRUE )



###############################
### Gráfico de Retornos + Volatilidade
plot( window(cbind(y, 10*fit@h.t),start=as.Date('2017-01-01')), main='Retornos e 100xVolatilidade', 
      multi.panel = TRUE)


### Gráfico de Retornos + VaR(95%)
VaR_95 <-  qsstd(0.05, mean=fit@fitted, sd = sqrt( fit@h.t ), nu=EMV["shape"], xi=EMV["skew"])
VaR_95 <- xts(VaR_95, order.by=index(y))

plot( window(cbind(y, VaR_95),start=as.Date('2017-01-01')), main='Retornos e VaR(95%)')

# taxa de cobertura
mean(y < VaR_95)


### Previsão da volatilidade
forec = predict(fit,n.ahead=252)
forec$standardDeviation^2 %>% plot()
#############################################################


