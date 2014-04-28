var SeriatedTermTopicProbabilityModel=Backbone.Model.extend({
	defaults:{
		matrix:null,
		termIndex:null,
		topicIndex:null,
		sparseMatrix:null,
		columnSums:null,
		topicMapping:null
	},
	url:"data/seriated-parameters.json",
	initialize:function(){this.parentModel=null}
});

SeriatedTermTopicProbabilityModel.prototype.initModel = function(b) {
	this.parentModel=filteredModel
};

SeriatedTermTopicProbabilityModel.prototype.load = function() {
	var b = function() {
		var b = [];
		var c = this.get("matrix");
		for(var a=0;a<this.get("topicIndex").length;a++) { 
			for(var d=0,e=0;e<this.get("termIndex").length;e++)
				d+=c[e][a];
			b.push(d);
		}
		this.set("columnSums",b)
	}.bind(this), 
	c = function(a,c,f) {
		b();
		this.printHTML("select");
		this.trigger("loaded:seriated")
	}.bind(this),
	
	a = function(a,b,c) {}.bind(this);

	this.fetch({
		add:!0,
		success:c,
		error:a
	})
};

SeriatedTermTopicProbabilityModel.prototype.printHTML = function(b) { 
	var c = this.get("termIndex").slice();
	c.sort();
	for(var a=0;a<c.length;a++)
		$(b).append("<option>"+escape(c[a])+"</option>");
	$(b).trigger("liszt:updated");
};
