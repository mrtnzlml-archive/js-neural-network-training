var neataptic = require('neataptic'); // this line is not needed in the browser
var fs = require('fs');

var Architect = neataptic.Architect;
var Methods = neataptic.Methods;
var Network = neataptic.Network;

//var myNetwork = new Architect.Perceptron(6,7,1);
var myNetwork = Network.fromJSON(JSON.parse(fs.readFileSync('network.json','utf8')));

var trainingSet = [
  { input: [0,0,0,0,0,0], output: [1] },
  { input: [1,0,0,0,0,0], output: [0] },
  { input: [0,0,0,1,0,0], output: [0] },
  { input: [1,0,0,1,0,0], output: [1] },
  // 2
  { input: [0,0,0,1,1,0], output: [0] },
  { input: [1,0,0,1,1,0], output: [0] },
  { input: [1,1,0,1,1,0], output: [1] },
  { input: [1,1,0,1,0,0], output: [0] },
  { input: [1,1,0,0,0,0], output: [0] },
  // 3
  { input: [0,0,0,1,1,1], output: [0] },
  { input: [1,0,0,1,1,1], output: [0] },
  { input: [1,1,0,1,1,1], output: [0] },
  { input: [1,1,1,1,1,1], output: [1] },
  { input: [1,1,1,1,1,0], output: [0] },
  { input: [1,1,1,1,0,0], output: [0] },
  { input: [1,1,1,0,0,0], output: [0] },
  // error fixing:
  { input: [1,0,0,0,0,0], output: [0] },
  { input: [0,1,0,0,0,0], output: [0] },
  { input: [0,0,1,0,0,0], output: [0] },
  { input: [0,0,0,1,0,0], output: [0] },
  { input: [0,0,0,0,1,0], output: [0] },
  { input: [0,0,0,0,0,1], output: [0] },
]


console.log(myNetwork.train(trainingSet, {
	rate: .1,
	iterations: 1e5,
	error: 1e-6,
	//shuffle: true,
	log: 1000,
	cost: Methods.Cost.CROSS_ENTROPY
}));


console.log('OK:')
echo(myNetwork.activate([0,0,0,0,0,0])); // 1
echo(myNetwork.activate([1,0,0,1,0,0])); // 1
echo(myNetwork.activate([1,1,0,1,1,0])); // 1

console.log('not OK:')
echo(myNetwork.activate([1,0,0,0,0,0])); // 0
echo(myNetwork.activate([0,1,0,0,0,0])); // 0
echo(myNetwork.activate([0,0,1,0,0,0])); // 0
echo(myNetwork.activate([0,0,0,1,0,0])); // 0
echo(myNetwork.activate([0,0,0,0,1,0])); // 0
echo(myNetwork.activate([0,0,0,0,0,1])); // 0

echo(myNetwork.activate([0,1,1,0,1,1])); // 0
echo(myNetwork.activate([0,1,0,0,1,0])); // 0
echo(myNetwork.activate([0,0,1,0,0,1])); // 0

function echo(value) {
  console.log(Math.round(value * 1000) / 1000)
}

fs.writeFile("network.json", JSON.stringify(myNetwork.toJSON(), null, 2), function(err) {
    if(err) {
        return console.log(err);
    }
    console.log("Neural network has been updated!");
});
