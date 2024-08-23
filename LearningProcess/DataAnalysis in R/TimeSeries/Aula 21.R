
#install.packages('quantmod')

require('quantmod') # pacote para financas




### carregando algumas series financeiras (observacoes diarias)

series <- new.env()               ## ambiente do R onde as series originais serao baixadas


fonte <- 'yahoo'                  ## fonte dos dados: banco de dados do yahoo
inicial <- as.Date('2000-01-01')  ## data inicial


## Indice Ibovespa --> Indice das principais acoes negociadas na bolsa Bovespa
getSymbols("^BVSP", env=series, src=fonte, from=inicial)

## Ativo PETR4 --> Acao da Petrobras
getSymbols("PETR4.SA", env=series, src=fonte, from=inicial)

## Ativo VALE3 --> Acao da Vale
getSymbols("VALE3.SA",  env=series, src=fonte, from=inicial)

## Ativo BBAS3 --> Acao do Banco do Brasil
getSymbols("BBAS3.SA",  env=series, src=fonte, from=inicial)







##### Estrutura dos dados ######

### series no formato OHLC, representando os valores diarios de:
# Open  ->  valor de abertura
# High  ->  maxima
# Low   ->  minima
# Close ->  fechamento

# mais duas colunas com os valores de 'volume de negociacoes' e 'valor de fechamento ajustado'.


### Exemplo -> series IBOVESPA

head(series$BVSP) # primeiras seis observacoes

tail(series$BVSP) # ultimas seis observacoes






############ plot ##############

x <- OHLC(series$BVSP) # filtra as colunas para OLHC

x %>% head()

par(mfrow=c(2,2))

plot( Op(x), main='Abertura diaria')    # grafico dos valores de abertura

plot( Hi(x), main='Maxima diaria')      # grafico dos valores de maximos diarios

plot( Lo(x), main='Minima diaria')      # grafico dos valores de minimaos diarios

plot( Cl(x), main='Fechamento diario')  # grafico dos fechamentos diarios



par(mfrow=c(1,1))

chart_Series( window(x,start=as.Date('2021-04-01'), end=as.Date('2021-04-30') ) )   # grafico de candlestick (grafico para representacao dos OHLC)






### Retornos diarios (exemplo: PETR4)

x <- Cl(series$PETR4.SA) ## fechamentos da PETR4

plot(x)

y <- diff( log(x) )      ## y <- log(x_t / x_{t-1}) ## retornos
# tambem pode-se usar a funcao dailyReturn(x, type='log')


## Sobre a series de retornos note que:

# 1) serie estacionaria com media igual a zero
par(mfrow=c(1,1)) #
plot(y, main='retornos da serie') 

# 2) aproximadamente sem autocorrelacoes
par(mfrow=c(2,1)) #
acf(na.omit(y), main='ACF dos retornos') 
pacf(na.omit(y), main='PACF dos retornos',ylim=c(-1,1)) 

# 3) Variancia autocorrelacionada
par(mfrow=c(2,1)) #
acf(na.omit(y^2), main='ACF dos retornos^2') 
pacf(na.omit(y^2), main='PACF dos retornos^2',ylim=c(-1,1)) 

# 4) Curtose > 3
timeDate::kurtosis( na.omit(y) , method='moment' )   
# curtose da normal
# timeDate::kurtosis( rnorm(1000) , method='moment' )


# 5) Assimetricos a esquerda
 z <- na.omit(y["::2019"])
 z <- (z-mean(z))/sd(z)
 par(mfrow=c(1,1))
 hist(z, prob=TRUE)
 lines(density(z))
 timeDate::skewness(z) ## Assimetria = \mu_3 / (\mu_2)^(3/2)
 
 
 
 
## Conclusoes sobre os retornos:
# 1) Possuem media condicional igual a zero: E_{t-1} [ y_t ] = 0
# 2) Entao: Var_{t-1} [ y_t ] = E_{t-1} [ y_t^2 ]
# 3) A presença de autocorrelacao em y^2 sugere que a variancia condicional nao constante
# 4) Tem curtose > 3 (distribuicao leptocurtica), ou seja, 
#     a distribuicao dos retornos apresenta caudas mais pesadas que a distribuicao normal 
# 5) Movimentos de baixa no mercado financeiro costumam ser mais rapidos que os movimentos alta,
#    isto reflete nos retornos, tornando-os assimetricos a esquerda (Coeficiente de assimetria < 0).






##### Correlacao em series de retornos #####
## Retornos tem media constante
## Assim podemos calcular correlacao, 
## obtendo assim um padrao medio da correlacao no periodo
## OBS: E de esperar que a correlacao entre series financeiras nao seja constante

y <- cbind( dailyReturn(series$BVSP, type='log'),
            dailyReturn(series$PETR4.SA, type='log'),
            dailyReturn(series$VALE3.SA, type='log'),
            dailyReturn(series$BBAS3.SA, type='log')
            )
names(y) <- c('BVSP','PETR4','VALE3','BBAS3')

y <- na.omit(y) ## limpa as linhas com NA

cor(y)          ## correlacao media no periodo
       
                ## correlacao media no periodo de 01/01/2020 ate 31/12/2020 
cor( window(y,start=as.Date('2020-01-01'),end=as.Date('2020-12-31')) )
