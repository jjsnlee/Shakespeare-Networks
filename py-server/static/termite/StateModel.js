var StateModel = Backbone.Model.extend({
	defaults : {
		"numAffinityTerms" : 25,
		"numSalientTerms" : 0,
		"totalTerms" : 25,
		"highlightedTerm" : "",
		"highlightedTopic" : null,
		"selectedTopics" : {},
		"doubleClickTopic": null,
		"selectedTopicsStr": ""	    // var for load and save state
	},
	initialize : function() {
		this.matrixView = null;
		this.termFreqView = null;	
	}
});

/**
 * Initialize state model's view models
 *
 * @private
 */
StateModel.prototype.initModel = function ( matrix, histogram ){
	this.matrixView = matrix;
	this.termFreqView = histogram;
};

/** 
 * Handles selecting topics using click event. Uses function freeColor and getColor that
 * are defined in ViewParameters
 *
 * @this {state model}
 * @param { string } DEFAULT defined in ViewParameters
 * @param { int } index of clicked topic
 */
StateModel.prototype.selectTopic = function( topicIndex ) {
	var color = DEFAULT;
	// frees the color associated with the topic if it was already selected
	if( topicIndex in this.get("selectedTopics")) {
		freeColor( this.get("selectedTopics")[topicIndex] );
		delete this.get("selectedTopics")[topicIndex];
	}
	// assign a color to the selected topic if there are any free 
	else {
	  // Should think about some way to pop the last color used
	
		color = getColor();
		this.get("selectedTopics")[topicIndex] = color;
	}
	// fire event to signify topic coloring may have changed
	this.trigger("color:topic", { "topic":topicIndex, "color": color } );
};

/**
 * Clears all topic selections (currently inefficiently implemented)
 */
StateModel.prototype.clearAllSelectedTopics = function() {
	console.log("clear all topics");
	var selectedTopics = this.get("selectedTopics");
	
	for(var i in selectedTopics) {
		freeColor( selectedTopics[i] );
		delete this.get("selectedTopics")[i];
		//if(i < topicIndex.length) {
		try {
			this.trigger("color:topic", {"topic":i, "color":DEFAULT} );
		}
		catch(err) {
			console.log('There was an error: '+i+': ['+err+']');
		}
	}
};


StateModel.prototype.setDoubleClickTopic = function ( topicIndex ){
	var type = "desc";
	this.set( "doubleClickTopic", topicIndex);
};
StateModel.prototype.clearSorting = function(){
	this.set( "doubleClickTopic", null);
};

/**
 * Handles highlighting events triggered by mouseover and mouseout
 * 
 * @param { string } target term
 * @param { int } index of target topic
 */
StateModel.prototype.setHighlightedTerm = function( term ) {
	this.set("highlightedTerm", term );
	//console.log( "Hello " + term + "!" );
};
StateModel.prototype.setHighlightedTopic = function( topic ) {
	this.set("highlightedTopic", topic );
};

StateModel.prototype.loadStatesFromQueryString = function() {
	var decodeString = function( str ){
		var topicLabel = "#topic:";
		var colorLabel = "#color:";
		// extract color and topic pairs
		while( str.length > 0) {
			var topicIndex = str.indexOf(topicLabel);
			var colorIndex = str.indexOf(colorLabel);

			var topic = null;
			var color = null;
			if(topicIndex >= 0 && colorIndex >= 0){
				topic = parseInt(str.substring(topicIndex+7, colorIndex));
				
				var tempIndex = str.indexOf(topicLabel, colorIndex+7);
				if(tempIndex >= 0){	//there's another pair
					color = str.substring(colorIndex+7, tempIndex);
					str = str.substring(tempIndex);
				} else { //no more pairs
					color = str.substring(colorIndex+7);
					// get rid of trailing characters...
					color = color.replace( /[^A-Za-z0-9]/g, "" );
					str = "";
				}
				this.get("selectedTopics")[topic] = color;
			}
		}
	}.bind(this);

	var qs = new QueryString();
	qs.addValueParameter( 'numAffinityTerms', 'na', 'int' );
	qs.addValueParameter( 'numSalientTerms', 'ns', 'int' );
	qs.addValueParameter( 'doubleClickTopic', 'dct', 'int');
	qs.addValueParameter( 'selectedTopicsStr', 'tc', 'str');
	
	var states = qs.read();
	for ( var key in states ){
		if(key === "doubleClickTopic" && states[key] === -1){
			this.set(key, null);
		}
		else if( key === "selectedTopicsStr" && states[key] !== ""){
			// decode string
			decodeString( states[key] );
			this.set(key, states[key]);
		}
		else if( key === "addTopTwenty"){
			if( states[key].replace( /[^A-Za-z0-9]/g, "" ) === "false")
				this.set(key, false);
			else
				this.set(key, true);
		}
		else
			this.set( key, states[key] );
	}

	this.trigger( "loaded:states" );
	this.trigger( "sending:colors", this.get("selectedTopics"));
};

StateModel.prototype.saveStatesToQueryString = function() {
	var qs = new QueryString();
	qs.addValueParameter( 'numAffinityTerms', 'na', 'int' );
	qs.addValueParameter( 'numSalientTerms', 'ns', 'int' );
	qs.addValueParameter( 'doubleClickTopic', 'dct', 'int');
	qs.addValueParameter( 'addTopTwenty', 'att', 'str');
	
	var selectedTopics = this.get("selectedTopics");
	var strVersion = "";
	for( var i in selectedTopics){
		if(selectedTopics[i] !== DEFAULT)
			strVersion += "#topic:" + i + "#color:" + selectedTopics[i];
	}
	this.set("selectedTopicsStr", strVersion);
	qs.addValueParameter( 'selectedTopicsStr', 'tc', 'str');
	
  // These have to be coordinated with the parameters listed above	
	var keys = [ 'numAffinityTerms', 'numSalientTerms', 
	               'doubleClickTopic', 'addTopTwenty', 'selectedTopicsStr' ];
	var states = {};
	for ( var i in keys )
	{
		var key = keys[i];
		if(key === "doubleClickTopic" && this.get(key) === null){
			states[key] = -1;
		}
		else
			states[key] = this.get(key);
	}
	
	qs.write( states );
};