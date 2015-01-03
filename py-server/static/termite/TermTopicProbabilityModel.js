/*
	TermTopicProbabilityModel.js
	
	This model is responsible for modifying data based on user inputs/controls 
		Current user control changes:
			-number of terms to show based on BEA choice order
			-number of terms to show based on saliency score (desc order)
			-specific terms to always show in the list of terms
			-whether or not to add top "twenty" terms of selected topics
			-sorting
	
	Details:
	--------
	Pulls data from SeriatedTermTopicProbabilityModel on initialize.
	Afterwards, this model is called when the user controls on the website are changed.
	At that time, the new "user defined" state is passed to the update function.  
*/

var TermTopicProbabilityModel = Backbone.Model.extend({
	defaults : {
		"matrix" : null,
		"termIndex" : null,
		"topicIndex" : null,
		"sparseMatrix" : null
	},
	//url : "../data/filtered-parameters.json",
	initialize : function() {
		this.stateModel = null;
		this.parentModel = null;
		
		this.termsPerTopic = 120;

		// mappings
		this.termRankMap = null;
		this.termOrderMap = null;
		//this.rowIndexMap = null;
		this.termDistinctivenessMap = null;
		this.termSaliencyList = [];
		
		// interaction related variables
		this.selectedTopics = {};
		this.visibleTopTerms = {};
	}
});

/**
 * Initialize filtered's parent and state model
 *
 * @private
 */
TermTopicProbabilityModel.prototype.initModel = function( model, state ){
	this.parentModel = model;
	this.stateModel = state;
};

/**
 * Initialize all topics' selection status to false (called once by load)
 *
 * @private
 */
TermTopicProbabilityModel.prototype.defaultSelection = function(){
	var topicIndex = this.parentModel.get("topicIndex");
	for( var i = 0; i < topicIndex.length; i++ ){
		this.selectedTopics[i] = false;
	}
};

/**
 * Loads various mappings from the model's "url"
 * and triggers a loaded event that the next model (child model) listens to.  
 * (This function is called after the seriated model loaded event is fired)
 *
 * @param { string } the location of datafile to load values from
 * @return { void }
 */
TermTopicProbabilityModel.prototype.load = function() {
//	var initRowIndexMap = function( termIndex ){
//		this.rowIndexMap = {};
//		for ( var i = 0; i < termIndex.length; i++ ){
//			this.rowIndexMap[termIndex[i]] = i;
//		}
//	}.bind(this);
	
	var initTermSaliencyList = function( saliencyMap ){
		termSaliencyList = [];
		tempList = [];
		for ( var term in saliencyMap ){
			tempList.push([term, saliencyMap[term]]);
		}
		tempList.sort(function(a, b) {return b[1] - a[1]});
		for( var i = 0; i < tempList.length; i++ ){
			this.termSaliencyList.push(tempList[i][0]);
		}
	}.bind(this);

	var successHandler = function( model, response, options ) {
		var keepQuiet = false;
		this.termRankMap = response.termRankMap;
		this.termOrderMap = response.termOrderMap;
		this.termDistinctivenessMap = response.termDistinctivenessMap;
		//initRowIndexMap( this.parentModel.get("termIndex") );
		initTermSaliencyList( response.termSaliencyMap );

		this.initTopTermLists();
		this.defaultSelection();
		this.filter( keepQuiet );	
		
		var coloredTopics = this.stateModel.get("selectedTopics");
		for( var obj in coloredTopics){
			claimColor( coloredTopics[obj] );
			this.selectTopic({"topic": obj, "color": coloredTopics[obj]} );
		}	
		
		this.trigger('loaded:filtered');

	}.bind(this);
	var errorHandler = function( model, xhr, options ) { }.bind(this);
	this.fetch({
		add : false,
		success : successHandler,
		error : errorHandler
	});
};

/** 
 * Generates list of top twenty terms per topic in original topicIndex (called in load)
 *
 * @private
 */
TermTopicProbabilityModel.prototype.initTopTermLists = function() {
	var termIndex = this.parentModel.get("termIndex");
	var topicIndex = this.parentModel.get("topicIndex");
	
	// This is only a very small subset NxT matrix, with T
	// being ~100 terms out of a vocabulary of 13k
	//var colFirstMatrix = generateColumnFirst(this.parentModel.get("matrix"));
	var topicTermFreqs = this.parentModel.get("matrix");
	
	var N = this.termsPerTopic;
	
	//var termsPerTopic = 20;	
	this.topTermLists = {};
	for( var i = 0; i < topicIndex.length; i++){
		// get term freqs for this topic
		var topicalFrequencies = topicTermFreqs[i];
		var topicTermScores = _.pairs(topicalFrequencies);
		topicTermScores.sort(function (a, b) { return a[1] < b[1] ? 1 : a[1] > b[1] ? -1 : 0; })
		topicTermScores = topicTermScores.slice(0, N);
		topicTermScores = _.filter(topicTermScores, function(v){ return v[1]>THRESHHOLD; });
		this.topTermLists[i] = _.map(topicTermScores, function(v){ return v[0]; });
	}
	
};

/**
 * Calls appropriate functions to update based on data change(s)
 */
TermTopicProbabilityModel.prototype.update = function( obj ) {
	this.filter( false );
};

/**
 * adds top twenty term list of selected topics to the visibleTopTerms list
 *
 * @private
 */
TermTopicProbabilityModel.prototype.addTopTerms = function() {
	for( var obj in this.selectedTopics){
		if(this.selectedTopics[obj])
			this.visibleTopTerms[obj] = this.topTermLists[obj];
	}
};

/**
 * Refreshes the termIndex and ordering based on user changes
 * 
 * @param { boolean } determines whether certain "set"s should trigger change events
 * @return { void }
 */
TermTopicProbabilityModel.prototype.filter = function( keepQuiet ) {
	var original_submatrix = this.parentModel.get("matrix");
	var original_termIndex = this.parentModel.get("termIndex");
	var original_topicIndex = this.parentModel.get("topicIndex");
	
	// For some reason this function is getting called twice every time
	// the page is refreshed!!
	
	var userDefinedTerms = this.stateModel.get("visibleTerms").slice(0);
	if(this.stateModel.get("addTopTwenty"))
		this.addTopTerms();
	else
		this.visibleTopTerms = {};
	
	var affinityLimit = this.stateModel.get("numAffinityTerms");
	var saliencyLimit = this.stateModel.get("numSalientTerms");
	
	var foundTerms = [];
	// choose terms to keep
	var chooseTerm = function(term){
		if(userDefinedTerms.indexOf(term) >= 0){
			foundTerms.push(term);
			return true;
		}
		if(this.termRankMap[term] < affinityLimit){
			return true;
		}
		if(this.termSaliencyList.indexOf(term) >= 0 
		      && this.termSaliencyList.indexOf(term) < saliencyLimit){
			return true;
		}
		for(var topicNo in this.visibleTopTerms){
			if(this.visibleTopTerms[topicNo].indexOf( term ) >= 0)
				return true;
		}
		return false;
	}.bind(this);
	
	// Basically using the Object to replicate a Set, to get unique values
	var termsToDisplay = _.object(original_termIndex, []);
	var addlTerms = _.map(this.visibleTopTerms, function(t) {
		return _.object(t, []);
	});

	termsToDisplay = _.reduce(addlTerms, function(memo, t) {
		return _.extend(memo, t)
	}, termsToDisplay);

	termsToDisplay = _.keys(termsToDisplay);

	var subset = [];
	
	// FIXME Need to test for cases where there is no topic selected
	var topic = this.stateModel.get("doubleClickTopic");
	if(topic) {
		// sort the terms
		for(var i=0; i<termsToDisplay.length; i++) {
			var term = termsToDisplay[i];
			if(chooseTerm( term ) ){
				if(topic > original_topicIndex.length) {
					topic = original_topicIndex.length-1;
				}
				
				//console.log('The currently selected topic is: '+topic);
				// Problem here when going from a larger dataset to a smaller one
				
				// Terms are going to be sorted by distinctiveness!
				// Is this across the entire corpus?
				subset.push( [term, 
				    original_submatrix[topic][term]
				        *this.termDistinctivenessMap[term]]);
			}
		}
	}

	// find out which user defined terms were found in the dataset
	for( var i = 0; i < foundTerms.length; i++){
		userDefinedTerms.splice(userDefinedTerms.indexOf(foundTerms[i]),1);
	}

	subset.sort(function(a, b) {
	   if(isNaN(a[1]) || isNaN(b[1])) {
	     if(!isNaN(a[1]))
	       return -1;
	     if(!isNaN(b[1]))
           return 1;
	     return 0;
	   } 
	   return b[1]-a[1];
	});

	// update model and state attributes
	matrix = [];	
	termIndex = []

	for(var i=0; i<subset.length; i++){
		var term = subset[i][0];
		termIndex.push(term);
		var termVals = [];
		for(var t=0; t<original_topicIndex.length; t++) {
		  termVals.push(original_submatrix[t][term])
		}
		matrix.push(termVals);
	}
	
	this.set("topicIndex", original_topicIndex, { silent: keepQuiet } );
	this.set("termIndex", termIndex, { silent: keepQuiet } );
	this.set("matrix", matrix, { silent: keepQuiet} );
	this.set("sparseMatrix", generateSparseMatrix.bind(this)(),  {silent: keepQuiet});
	
	this.stateModel.setFoundTerms(foundTerms, keepQuiet);
	this.stateModel.setUnfoundTerms(userDefinedTerms, keepQuiet);
	this.stateModel.set("totalTerms", termIndex.length);
};

/**
 * Behavior when topic is selected
 *
 * @this { TermTopicProbabilityModel }
 * @param { object } topic: target topic index, color: associated color
 * @return { void }
 */
TermTopicProbabilityModel.prototype.selectTopic = function( obj ) {
	var topic = obj.topic;
	var colorClass = obj.color;
	var topicIndex = this.parentModel.get("topicIndex");
	if( topic !== null){

		// if color is DEFAULT, the event can be treated as a deselect
		if( colorClass === DEFAULT){
			if(this.selectedTopics[topic]){
				delete this.visibleTopTerms[topic]; 
				this.selectedTopics[topic] = false;
				this.filter( false );
			}
			return;
		}
			
		// only add if this topic wasn't added previously
		if(this.selectedTopics[topic] === false) {
			this.selectedTopics[topic] = true;
			this.filter( false );
		}
	}
};

/**
 * Generates a descending sorted sparse matrix representation of a full matrix. 
 * Must be called by a model that has termIndex, topicIndex, and matrix default vars
 * (e.g. seriated model, filtered model)
 *
 * @this { a termTopic model }
 * @param { double } THRESHHOLD is defined in ViewParameters
 * @param { array } termIndex is a list of ordered terms, size n
 * @param { array } topicIndex is a list of ordered topics, size m
 * @param { 2D array } matrix is a n x m matrix of doubles
 * @return { array } Sparse matrix representation of matrix (list of objects)
 */
var generateSparseMatrix = function() {
  var termIndex = this.get("termIndex");
  var topicIndex = this.get("topicIndex");
  var matrix = this.get("matrix");
  var sparseMatrix = [];
  for ( var i = 0; i < termIndex.length; i++ ) {
    for ( var j = 0; j < topicIndex.length; j++ ) {
      if ( matrix[i][j] > THRESHHOLD ) {
        sparseMatrix.push({
          'term' : termIndex[i],
          'termIndex' : i,
          'topicName' : topicIndex[j],
          'topicIndex' : j,
          'value' : matrix[i][j]
          });
      }
    }
  }
  sparseMatrix = sparseMatrix.sort( function(a,b) { return b.value - a.value } );
  return sparseMatrix;
};

/**
 * Returns a column-first representation of a row-first matrix
 * 
 * @this { index.html }
 * @param { 2D array } row first matrix
 * @return { 2D array } column first matrix
 */
var generateColumnFirst = function( matrix ) {
    //var matrix = this.get("matrix");
    var colMatrix = [];
    if(matrix === null)
        return null;
    // init empty rows of column matrix
    for (i = 0; i < matrix[0].length; ++i) {
        colMatrix.push([]);
    }
    // fill in values
    for (i = 0; i < matrix.length; ++i) {
        var row = matrix[i];
        for (j = 0; j < row.length; ++j) {
            colMatrix[j].push(row[j]);
        }
    }
    return colMatrix;
};
