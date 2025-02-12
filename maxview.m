function max_view = maxview(A,B,weightAB,halfAB)
%%%Output the maximum indicator value under a certain indicator, the corresponding standard deviation, and the ID of the corresponding perspective.
    [mean_a,mean_b,mean_w,mean_h] = deal(nanmean(A),nanmean(B),nanmean(weightAB),nanmean(halfAB));  
    [std_a,std_b,std_w,std_h] = deal(nanstd(A),nanstd(B),nanstd(weightAB),nanstd(halfAB)); 
    std = [std_a,std_b,std_w,std_h];
        
    [max_allview,view_id] = max([mean_a,mean_b,mean_w,mean_h]);   %Correspondence between ID and perspective
    std_allview = std(view_id);
    max_view.value = max_allview;
    max_view.std = std_allview;
    max_view.id = view_id;
end
    