
library(magrittr)



#################################
###########     STL     #########
#################################

x <- AirPassengers


##### variando s.window ########
stl(x, s.window=3) %>% plot( main='s.window=3' )

stl(x, s.window=5) %>% plot( main='s.window=5' )

stl(x, s.window=7) %>% plot( main='s.window=7' )

stl(x, s.window=9) %>% plot( main='s.window=9' )

stl(x, s.window=11) %>% plot( main='s.window=11' )

stl(x, s.window=13) %>% plot( main='s.window=13' )

stl(x, s.window='periodic') %>% plot( main='s.window=periodic' )




##########################################
### Multiplicativa usando log     ########
##########################################

logx <- log(x)

plot(logx)

stl(logx, s.window='periodic') %>% plot( main='log(x)' )

stl(logx, s.window=3) %>% plot( main='s.window=3' )

stl(logx, s.window=5) %>% plot( main='s.window=5' )

stl(logx, s.window=7) %>% plot( main='s.window=7' )

stl(logx, s.window=9) %>% plot( main='s.window=9' )

stl(logx, s.window=11) %>% plot( main='s.window=11' )

stl(logx, s.window=13) %>% plot( main='s.window=13' )
























##########################################
### função mstl() do pacote forecast  ####
##########################################

require(forecast)

x %>% mstl(lambda = NULL) %>% plot

x %>% mstl(lambda = "auto") %>% plot




library(ggplot2)

plot(taylor)

mstl(taylor) %>% autoplot(facet=TRUE)

