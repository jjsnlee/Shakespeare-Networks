var FilteredTermTopicProbabilityModel=Backbone.Model.extend({
	defaults:{
		matrix:null,
		termIndex:null,
		topicIndex:null,
		sparseMatrix:null,
		normalizedSparseMatrix:null
	},
	url:"data/filtered-parameters.json",
	initialize:function(){
		this.termDistinctivenessMap=
			this.rowIndexMap =
			this.termOrderMap =
			this.termRankMap =
			this.parentModel =
			this.stateModel = null;
		
		this.termSaliencyList=[];
		this.visibleTopTerms={}
	}
});

FilteredTermTopicProbabilityModel.prototype.initModel = function(b,e) {
	// SeriatedTermProbModel
	this.parentModel=b;
	// StateModel
	this.stateModel=e
};

FilteredTermTopicProbabilityModel.prototype.load=function(){
	var b=function(c){this.rowIndexMap={};for(var a=0;a<c.length;a++)this.rowIndexMap[c[a]]=a}.bind(this),e=function(c){termSaliencyList=[];tempList=[];for(var a in c)tempList.push([a,c[a]]);tempList.sort(function(a,c){return c[1]-a[1]});for(c=0;c<tempList.length;c++)this.termSaliencyList.push(tempList[c][0])}.bind(this),h=function(c,a,d){this.termRankMap=a.termRankMap;this.termOrderMap=a.termOrderMap;
	this.termDistinctivenessMap=a.termDistinctivenessMap;
	b(this.parentModel.get("termIndex"));e(a.termSaliencyMap);this.initTopTermLists();this.filter();this.trigger("loaded:filtered")}.bind(this),g=function(c,a,d){}.bind(this);
	this.fetch({add:!1,success:h,error:g})
};

FilteredTermTopicProbabilityModel.prototype.initTopTermLists = function() {
	var b = this.parentModel.get("termIndex"),
	e = this.stateModel.get("topicIndex"),
	h=generateColumnFirst(this.parentModel.get("matrix"));
	
	this.topTermLists={};
	for(var g=0;g<e.length;g++) { 
		this.topTermLists[g]=[];
		for(var c=h[g],a=Array(b.length),d=0;d<b.length;d++)
			a[d]=d;
		a.sort(function(a,d){return c[a]<c[d]?1:c[a]>c[d]?-1:0});
		for(d=0; 20>d && a[d]>THRESHHOLD;)
			this.topTermLists[g].push(b[a[d]]),d++
	}
};

FilteredTermTopicProbabilityModel.prototype.update = function() {
	this.filter()
};

FilteredTermTopicProbabilityModel.prototype.addTopTerms=function(){
	var b=_.groupBy(stateModel.get("topicIndex"),"selected")["true"];
	this.visibleTopTerms={};
	for(var e in b) { 
		var h=b[e];
		this.visibleTopTerms[h.id] = this.topTermLists[h.id]
	}
};

FilteredTermTopicProbabilityModel.prototype.filter=function() {
	var b=this.parentModel.get("matrix"),
	e=this.parentModel.get("termIndex"),
	h=this.parentModel.get("topicIndex"),
	g=this.stateModel.get("visibleTerms").slice(0);
	
	this.stateModel.get("addTopTwenty")?this.addTopTerms():this.visibleTopTerms={};
	
	var c=this.stateModel.get("numAffinityTerms"),
	a=this.stateModel.get("numSalientTerms"),
	d=[],
	l=[],
	p = function(b) {
		if(0<=g.indexOf(b)) return d.push(b),!0;
		if(this.termRankMap[b]<c||0<=this.termSaliencyList.indexOf(b)&&this.termSaliencyList.indexOf(b)<a)
			return!0;
		for(var e in this.visibleTopTerms)
			if(0<=this.visibleTopTerms[e].indexOf(b))
				return!0;
		return!1
	}.bind(this),
	m=this.stateModel.get("sortType");
	
	for(var k=0; k<e.length; k++) {
		var f=e[k];
		if(p(f))
			if(""===m)
				l.push([f,this.termOrderMap[f]]);
			else if("desc"===m) {
				var n=this.stateModel.get("doubleClickTopic");
				l.push([f,1/(b[this.rowIndexMap[f]][n]*this.termDistinctivenessMap[f])])
			}
			else 
				"asc"===m && (n=this.stateModel.get("doubleClickTopic"),
					l.push([f,b[this.rowIndexMap[f]][n]*this.termDistinctivenessMap[f]]) )
	}

	for(k=0;k<d.length;k++)
		g.splice(g.indexOf(d[k]),1);

	l.sort(function(a,b) {return a[1]-b[1]});
	matrix=[];
	termIndex=[];
	for(e=0;e<l.length;e++)
		f=l[e][0],termIndex.push(f),matrix.push(b[this.rowIndexMap[f]]);

	this.set("topicIndex",h);
	this.set("termIndex",termIndex);
	this.set("matrix",matrix);
	this.set("normalizedSparseMatrix",generateSparseMatrix.bind(this)(this.parentModel.get("columnSums")));
	this.set("sparseMatrix",generateSparseMatrix.bind(this)(null));
	this.stateModel.set("totalTerms",termIndex.length)
};
