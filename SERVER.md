# SERVER MANAGEMENT
This is the management guide for GAIA server.

## Welcome message
Install figlet
```bash
$ sudo apt-get install figlet
```

```bash
$ sudo vi /etc/bash.bashrc

# Press [Shift] + [G] and write the code on the bottom
clear
printf "Welcome to Ubuntu 16.04.5 LTS (GNU/Linux-Mint-18 x86_64)\n"
printf "This is the server for the Digital Mammography DREAM Challenge team.\n\n"
printf " * Documentation: https://github.com/meliketoy/DreamChallenge\n\n"
printf "The challenge is supported until March 2017.\n"
printf "##############################################################\n"
figlet -f slant "gaia dmis"
printf "\n\n"
printf " Data Mining & Information System Lab\n"
printf " GPU Computing machine : GAIA@163.152.163.152\n\n"
printf " Administrator   : Bumsoo Kim\n"
printf " Administrator   : Minhwan Yu\n"
printf " Administartor   : Hwejin Jung\n\n"
printf " Please read the document\n"
printf "    https://github.com/meliketoy/GAIA-server-policy/README.md\n"
printf "##############################################################\n"
```

## Remote Server control
```bash
$ sudo vi ~/.bashrc

# Press [Shift] + [G] and enter the function on the bottom.

function server_func() {
    echo -n "[Enter the name of server]: "
    read server_name
                
    # echo "Logging in to server $server_name ..."
    if [ $server_name == [:servername] ]; then
        ssh [:usr]@[:ip].[:ip].[:ip].[:ip]
    fi
}
alias dmis_remote=server_func
```

## Github control

```bash
$ sudo vi ~/.netrc

machine github.com
login [:username]
password [:password]
```
