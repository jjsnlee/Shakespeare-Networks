var StateModel = Backbone.Model.extend({
	defaults: { 
		numAffinityTerms:50,
		numSalientTerms:0,
		visibleTerms:[],
		totalTerms:50,
		sortType:"",
		normColumns:!1,
		addTopTwenty:!1,
		highlightedTerm:"",
		highlightedTopic:null,
		doubleClickTopic:null,
		topicIndex:[],
		version:0
	},
	initialize : function() { 
		this.colorNames="orange blue green purple brown pink".split(" ");
		this.colorObjs=[];
		this.justAddedTerm=null;
		for(var a=0;a<this.colorNames.length;a++)
			this.colorObjs.push({color:this.colorNames[a],usage:!1})
	},
	// wonder what format this json needs to be? 
	url:"state.json"
});

StateModel.prototype.getColor=function(){for(var a=DEFAULT,b=0;b<this.colorObjs.length;b++)if(!this.colorObjs[b].usage){a=this.colorObjs[b].color;this.colorObjs[b].usage=!0;break}return a};
StateModel.prototype.colorUsage=function(a,b){if(a!==DEFAULT)for(var c=0;c<this.colorObjs.length;c++)if(a===this.colorObjs[c].color){this.colorObjs[c].usage=b;break}};
StateModel.prototype.batchClaimColors=function(){var a=_.groupBy(this.get("topicIndex"),"selected")["true"];if(void 0!==a)for(var b in a)this.colorUsage(a[b].color,!0)};
StateModel.prototype.setVisibleTerms=function(a){var b=_.difference(a,this.get("visibleTerms"));this.justAddedTerm=0===b.length?null:b[0];this.set("visibleTerms",a)};
StateModel.prototype.getJustAddedTerm=function(){return this.justAddedTerm};
StateModel.prototype.selectTopic=function(a){var b=_.object(_.map(this.get("topicIndex"),function(a){return[a.id,a]})),c=DEFAULT;b[a].color!==DEFAULT?(this.colorUsage(b[a].color,!1),this.get("topicIndex")[b[a].position].color=DEFAULT,this.get("topicIndex")[b[a].position].selected=!1):(c=this.getColor(),this.get("topicIndex")[b[a].position].color=c,this.get("topicIndex")[b[a].position].selected=!0);this.trigger("color:topic",{topic:a,color:c})};
StateModel.prototype.clearAllSelectedTopics=function(){var a=this.get("topicIndex"),b;for(b in a)a[b].color!==DEFAULT&&(this.colorUsage(a[b].color,!1),a[b].color=DEFAULT,a[b].selected=!1,this.trigger("color:topic",{topic:a[b].id,color:DEFAULT}));this.set("topicIndex",a);this.trigger("clear:allTopics")};
StateModel.prototype.getSortType=function(a){var b=["desc","asc",""];if(this.get("doubleClickTopic")!==a)return b[0];a=this.get("sortType");a=(b.indexOf(a)+1)%b.length;return b[a]};
StateModel.prototype.setDoubleClickTopic=function(a){var b=this.getSortType(a);""===b?this.set("doubleClickTopic",null):this.set("doubleClickTopic",a);this.set("sortType",b)};
StateModel.prototype.clearSorting=function(){this.set("doubleClickTopic",null);this.set("sortType","")};
StateModel.prototype.setHighlightedTerm=function(a){this.set("highlightedTerm",a)};
StateModel.prototype.setHighlightedTopic=function(a){this.set("highlightedTopic",a)};

StateModel.prototype.updateLabel = function(a,b) {
	this.get("topicIndex")[a]!==b&&(this.get("topicIndex")[a].name=b,this.trigger("change:topicLabels"))
};

StateModel.prototype.moveTopic=function(a,b) {
	console.log("user wants to move topic "+a+" to "+b);
	a=parseInt(a);b=parseInt(b);
	var c=this.get("topicIndex");
	if(a<b)
		for(var d=a+1;d<=b;d++)c[d].position-=1;
	else if(a>b)
		for(d=b;d<a;d++)
			c[d].position+=1;
	c[a].position=b;
	c.sort(function(a,b){return a.position-b.position});
	console.log(c);
	this.trigger("change:topicPosition")
};

StateModel.prototype.originalPositions = function() {
	for(var a=this.get("topicIndex"),b=0;b<a.length;b++)
		a[b].position=a[b].id;
	a.sort(function(a,b) {
		return a.position-b.position
	});
	this.trigger("change:topicPosition")
};

var loadStatefromDB = function(dbg) {
	if(dbg) console.log("initial load from db");
	stateModel.clearAllSelectedTopics();
	stateModel.fetch({
		success : function() {
			console.log("loaded state from db");
			stateModel.batchClaimColors();
			if(dbg) stateModel.trigger("loaded:states")
		}
	})
};
