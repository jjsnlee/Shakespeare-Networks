var TERMFREQ_TEXT_DEFAULT={FILL_COLOR:"#808080",STROKE_OPACITY:0,FILL_OPACITY:1},
TERMFREQ_BAR_DEFAULT={STROKE_COLOR:"#808080",STROKE_WIDTH:5,STROKE_OPACITY:0.4},
HISTOGRAM_ENCODING_PARAMETERS={NUM_TOPICS:0,setNumTopics:function(a){
	this.NUM_TOPICS=a
},
DENSE_NUM_TOPICS:50,
LOOSE_NUM_TOPICS:20,
DENSE_PACKING:12,
LOOSE_PACKING:18,
packing:function(){return 12}},
HISTORGRAM_CONTAINER_PADDING={left_separation:10,top:105,left:130,right:20,bottom:60,width:150,
	fullWidth:function(){return this.left+this.right+this.width},
	fullHeight:function(a,b){
		return this.top+this.bottom+HISTOGRAM_ENCODING_PARAMETERS.packing()*b
	}
},
TermFrequencyView=Backbone.View.extend({
	initialize:function(){
		this.highlightedTopic=this.highlightedTerm=this.svgTermHighlightLayer=
			this.svgTopicalBarLayer=this.overlayLineLayer=this.overlayLayer=
			this.svgTermBarLayer=this.svgTermLabelLayer=this.svg=this.line_length=
			this.ys=this.stateModel=this.parentModel=null;
		this.colorClassPrefix="HIST";
		this.normalColor="normal";
		this.totalOffsets=[];
		this.prevHighlightColor=this.normalColor;this.useOffset=!1}
});

TermFrequencyView.prototype.initModel=function(a,b){
	this.parentModel=a;this.stateModel=b
};

TermFrequencyView.prototype.getSelectedTopics=function(){
	var a=_.groupBy(this.stateModel.get("topicIndex"),"selected")["true"];return void 0!==a?a:[]
};

TermFrequencyView.prototype.load=function(){this.renderInit();this.renderUpdate()};
TermFrequencyView.prototype.update=function(){this.renderUpdate()};
TermFrequencyView.prototype.prepareStackedBars=function(){
	var a=this.parentModel.get("topicalFreqMatrix"),b=[];
	0<a.length&&(b=a.map(function(a){
		return a.map(function(a,g){return{x:g,y:a}})
	}),b=d3.layout.stack()(b));
	a=_.object(_.map(this.stateModel.get("topicIndex"),function(a){
		return[a.id,a.selected]
	}));
	this.totalOffsets=[];
	if(1<b.length||!1===a[this.highlightedTopic]&&1==b.length)
		for(a=0;a<b[0].length;a++){
			for(var c=0,d=0;d<b.length;d++)c+=b[d][a].y;
			this.totalOffsets[a]=c}
	else 
		for(a=0;a<this.parentModel.get("termIndex").length;a++)
			this.totalOffsets[a]=0;
	return b
};

TermFrequencyView.prototype.renderInit=function(){
	var a=this.parentModel.get("termIndex"),b=this.parentModel.get("totalTermFreqs");
	this.ys=d3.scale.linear();
	for(var c=0,d=0;d<a.length;d++)
		b[a[d]]>c&&(c=b[a[d]]);
	
	this.line_length=d3.scale.linear().domain([0,c]).range([0,HISTORGRAM_CONTAINER_PADDING.width]);

	this.svg =
		d3.select(this.el).
		append("svg:svg").
		style("cursor","default").
		style("width",HISTORGRAM_CONTAINER_PADDING.fullWidth()+"px");
	
	this.svgTermLabelLayer =
		this.svg.append("svg:g").
		attr("class","termLabelLayer").
		attr("transform","translate("+HISTORGRAM_CONTAINER_PADDING.left+","+HISTORGRAM_CONTAINER_PADDING.top+")");

	this.svgTermBarLayer=
		this.svg.append("svg:g").
		attr("class","termBarLayer").
		attr("transform","translate("+HISTORGRAM_CONTAINER_PADDING.left+","+HISTORGRAM_CONTAINER_PADDING.top+")");
	
	this.overlayLayer=
		this.svg.append("svg:g").
		attr("class","overlayLayer").
		attr("transform","translate("+HISTORGRAM_CONTAINER_PADDING.left+","+HISTORGRAM_CONTAINER_PADDING.top+")");
	
	this.svgTopicalBarLayer=
		this.svg.append("svg:g").
		attr("class","topicalBarLayer").
		attr("transform","translate("+HISTORGRAM_CONTAINER_PADDING.left+","+HISTORGRAM_CONTAINER_PADDING.top+")");
	
	this.svgTermHighlightLayer=
		this.svg.append("svg:g").
		attr("class","termHighlightLayer").
		attr("transform","translate("+HISTORGRAM_CONTAINER_PADDING.left+","+HISTORGRAM_CONTAINER_PADDING.top+")");
	
	MAIN_WIDTH.TermFrequencyView=HISTORGRAM_CONTAINER_PADDING.fullWidth();updateMainWidth()
};

TermFrequencyView.prototype.renderUpdate=function(){
	var a=this.parentModel.get("termIndex"),b=this.parentModel.get("totalTermFreqs");
	this.svg.style("height",HISTORGRAM_CONTAINER_PADDING.fullHeight(HISTOGRAM_ENCODING_PARAMETERS.NUM_TOPICS,a.length)+"px")
	this.ys.domain([0,a.length]).range([0,a.length*HISTOGRAM_ENCODING_PARAMETERS.packing()]);
	this.svgTermLabelLayer.selectAll("text").data(a).exit().remove();

	this.svgTermLabelLayer.selectAll("text").data(a).enter().append("svg:text").on("mouseout",function(){
			this.trigger("mouseout:term","")
		}.bind(this)).attr("x",-HISTORGRAM_CONTAINER_PADDING.left_separation).attr("y",3);
	
	this.svgTermLabelLayer.selectAll("text").data(a).attr("class",function(a){
			return["termLabel HISTnormal",getTermClassTag(a)].join(" ")
		}).attr("transform",function(a,b){
			return"translate(0,"+this.ys(b+0.5)+")"
		}.bind(this)).on("mouseover",function(a){
			this.trigger("mouseover:term",a)
		}.bind(this)).text(function(a){return a});
	
	this.svgTermBarLayer.selectAll("line").data(a).exit().remove();
	this.svgTermBarLayer.selectAll("line").data(a).enter().append("svg:line").on("mouseout",function(){
			this.trigger("mouseout:term","")
		}.bind(this)).attr("y1",0).attr("y2",0).attr("x1",this.line_length(0));
	
	this.svgTermBarLayer.selectAll("line").data(a).attr("transform",function(a,b){
			return"translate(0,"+this.ys(b+0.5)+")"
		}.bind(this)).attr("class",function(a,b){
			return["termFreqBar",getTermClassTag(a)].join(" ")
		}).on("mouseover",function(a){
			this.trigger("mouseover:term",a)
		}.bind(this)).attr("x2",function(a){return this.line_length(b[a])}.bind(this));
	
	var c=this.prepareStackedBars(),d=this.getSelectedTopics();
	this.overlayLayer.selectAll("g").data(c).exit().remove();
	this.overlayLayer.selectAll("g").data(c).enter().append("svg:g");

	this.gLayer = 
		this.overlayLayer.
		selectAll("g").data(c).
		attr("class",function(a,b){
			return["overlayGroup",this.colorClassPrefix+d[b].color].join(" ")
		}.bind(this));

	this.gLayer.selectAll("line").data(function(a){return a}).exit().remove();

	this.gLayer.selectAll("line").data(function(a,b){return a}).enter().append("svg:line").on("mouseout",function(){
			this.trigger("mouseout:term","")
		}.bind(this)).attr("y1",0).attr("y2",0);

	this.gLayer.selectAll("line").data(function(a,b){return a}).
		attr("class", function(b,e){
			return["line",getTermClassTag(a[e])].join(" ")
		}).on("mouseover",function(b,e){
			this.trigger("mouseover:term",a[e])
		}.bind(this)).attr("transform", function(a,b){
			return"translate(0,"+this.ys(b+0.5)+")"
		}.bind(this)).attr("x1",function(a){
			return this.line_length(a.y0)
		}.bind(this)).attr("x2",function(a){
			return this.line_length(a.y0)+this.line_length(a.y)
		}.bind(this));
	
	this.svgTopicalBarLayer.selectAll("line").data(a).exit().remove();
	this.svgTopicalBarLayer.
		selectAll("line").
		data(a).enter().append("svg:line").
		on("mouseout",function(){
			this.trigger("mouseout:term","")
		}.bind(this)).
		attr("y1",0).
		attr("y2",0).
		attr("x1", this.line_length(0)).
		attr("x2",this.line_length(0));
	
	this.svgTopicalBarLayer.selectAll("line").data(a).attr("transform",function(a,b){
		return"translate(0,"+this.ys(b+0.5)+")"}.bind(this)).attr("class",function(a,b){
			return["topicalFreqBar",getTermClassTag(a)].join(" ")
		}).on("mouseover",function(a){
			this.trigger("mouseover:term",a)
		}.bind(this));
	
	this.svgTermHighlightLayer.selectAll("line").data(a).exit().remove();
	
	this.svgTermHighlightLayer.selectAll("line").data(a).enter().append("svg:line").on("mouseout",function(){
			this.trigger("mouseout:term","")
		}.bind(this)).attr("y1",0).attr("y2",0).attr("x1",this.line_length(0)).style("fill","none");

	this.svgTermHighlightLayer.selectAll("line").data(a).attr("transform",function(a,b){
			return"translate(0,"+this.ys(b+0.5)+")"
		}.bind(this)).attr("class",function(a,b){
			return["termHighlightBar",getTermClassTag(a)].join(" ")
		}).on("mouseover", function(a){this.trigger("mouseover:term",a)}.bind(this)).attr("x2", function(a){
			return this.line_length(b[a])
		}.bind(this))
};

TermFrequencyView.prototype.onHighlightTopicChanged=function(a,b){
	null===b?this.unhighlight(!1,!0):this.highlight(null,b)
};

TermFrequencyView.prototype.onHighlightTermChanged=function(a,b){
	""===b?this.unhighlight(!0,!1):this.highlight(b,null)
};

TermFrequencyView.prototype.unhighlight=function(a,b){
	a&&(a=this.highlightedTerm,this.highlightedTerm=null,this.svgTermLabelLayer.selectAll("."+getTermClassTag(a)).classed(this.colorClassPrefix+HIGHLIGHT,!1),this.svgTermHighlightLayer.selectAll("."+getTermClassTag(a)).classed(this.colorClassPrefix+HIGHLIGHT,!1));
	if(b){
		b=this.highlightedTopic;
		var c=this.parentModel.get("termIndex"),d=this.parentModel.getTopicalsForTopic(b);
		this.highlightedTopic=null;
		for(var g=0;g<c.length;g++)
			a=c[g],d[g]>THRESHHOLD&&(this.svgTermLabelLayer.selectAll("."+getTermClassTag(a)).classed(this.colorClassPrefix+HIGHLIGHT,!1),this.useOffset&&this.svgTopicalBarLayer.selectAll("."+getTermClassTag(a)).classed(this.colorClassPrefix+HIGHLIGHT,!1).attr("x2",this.line_length(0)).attr("x1",this.line_length(0)));
		var e=this.getSelectedTopics();
		
		this.gLayer=0<e.length?this.overlayLayer.selectAll("g").attr("class",function(a,b){
			return["overlayGroup",this.colorClassPrefix+e[b].color].join(" ")
		}.bind(this)):this.overlayLayer.selectAll("g").attr("class",function(a,b){
			return["overlayGroup",this.colorClassPrefix+this.normalColor].join(" ")
		}.bind(this));this.prevHighlightColor=this.normalColor;this.useOffset=!1
	}
};

TermFrequencyView.prototype.highlight=function(a,b){
	if(null!==a)
		this.highlightedTerm=a,this.svgTermLabelLayer.selectAll("."+getTermClassTag(a)).classed(this.colorClassPrefix+HIGHLIGHT,!0),this.svgTermHighlightLayer.selectAll("."+getTermClassTag(a)).classed(this.colorClassPrefix+HIGHLIGHT,!0);
	else if(null!==b){
		var c=this.parentModel.get("termIndex"),d=this.parentModel.getTopicalsForTopic(b);
		this.highlightedTopic=b;
		this.prepareStackedBars();
		for(var g=[],e=this.getSelectedTopics(),f=0;f<e.length;f++)
			g.push(e[f].color);
		f=_.object(_.map(this.stateModel.get("topicIndex"),function(a){
			return[a.id,a]
		}));
		f[b].selected?(this.prevHighlightColor=f[b].color,g[g.indexOf(this.prevHighlightColor)]=HIGHLIGHT,
				this.gLayer=this.overlayLayer.selectAll("g").attr("class",
						function(a,b){
							return["overlayGroup",this.colorClassPrefix+g[b]].join(" ")
							}.bind(this)))
					: this.useOffset=!0;

		for(f=0;f<c.length;f++)
			a=c[f],d[f]>THRESHHOLD&&
			(this.svgTermLabelLayer.selectAll("."+getTermClassTag(a)).classed(this.colorClassPrefix+HIGHLIGHT,!0),
					this.useOffset&&(e=0,0<this.totalOffsets.length
							&&(e=this.totalOffsets[f]),this.svgTopicalBarLayer.selectAll("."+getTermClassTag(a)).
							classed(this.colorClassPrefix+HIGHLIGHT,!0).
							attr("x2",this.line_length(e+d[f])).attr("x1",this.line_length(e))))}
};
