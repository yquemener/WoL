#!/bin/bash

python server.py > /dev/null &
serverPID=$!
echo $serverPID
sleep 1
python client_nettest.py 0 > /dev/null &
client1PID=$!
echo $client1PID
python client_nettest.py 800 > /dev/null &
client2PID=$!
echo $client2PID
wait -n $client1PID $client2PID

echo "Killing server"
kill $serverPID
kill $client1PID
kill $client2PID
