var TermFrequencyModel=Backbone.Model.extend({
	defaults:{
		termIndex:null,
		totalTermFreqs:{},
		topicalFreqMatrix:[],
		selectedTopics:{}
	},
	url:"data/global-term-freqs.json",
	initialize:function() { 
		this.termFreqMap=
			this.topicMapping=
			this.originalTermIndex=
			this.originalTopicIndex=
			this.originalMatrix=
			this.stateModel=
			this.parentModel=null
	}
});

TermFrequencyModel.prototype.initModels = function(c,a) {
	this.parentModel=c;this.stateModel=a
};

TermFrequencyModel.prototype.load=function(){
	var c=function(a,b,c){this.set("termIndex",this.parentModel.get("termIndex"));
	this.originalMatrix=b.matrix;
	this.originalTopicIndex=b.topicIndex;
	this.originalTermIndex=b.termIndex;
	this.topicMapping=b.topicMapping;
	this.termFreqMap=b.termFreqMap;
	this.getTotalTermFreqs();
	void 0!==_.groupBy(this.stateModel.get("topicIndex"),"selected")["true"] && this.update();
	this.trigger("loaded:freqModel")}.bind(this),
	a=function(a,b,c){}.bind(this);
	
	this.fetch({add:!1,success:c, error:a})
};

TermFrequencyModel.prototype.update = function() {
	this.set("termIndex",this.parentModel.get("termIndex"));
	this.generateTopicalMatrix(!1);this.trigger("updated:TFM")
};

TermFrequencyModel.prototype.getTotalTermFreqs = function() {
	for(var c={},a=this.parentModel.get("termIndex"),d=0;d<a.length;d++)
		c[a[d]]=this.termFreqMap[a[d]];
	this.set("totalTermFreqs",c)
};

TermFrequencyModel.prototype.generateTopicalMatrix = function(c) {
	var a=[],d=this.parentModel.get("termIndex"),
	b=_.groupBy(this.stateModel.get("topicIndex"),"selected")["true"];
	if(void 0!==b)
		for(var e=0;e<b.length;e++){
			for(var g=[],h=b[e].id,f=0;f<d.length;f++){
				var k=this.originalTermIndex.indexOf(d[f]);
				g.push(this.originalMatrix[k][h])
			}
			a.push(g)
		}
	this.getTotalTermFreqs();
	this.set("topicalFreqMatrix", a, {silent:c});
	return a
};

TermFrequencyModel.prototype.getTopicalsForTopic = function(c){
	for(var a=[],d=this.get("termIndex"),b=0;b<d.length;b++){
		var e = this.originalTermIndex.indexOf(d[b]);
		a.push(this.originalMatrix[e][c])
	}
	return a
};
