
const Dashboard = () => {

  return (
    <div>
      <h1>Dashboard</h1>

      <iframe src="http://localhost:5601/app/dashboards#/view/1c4bcb30-01e2-11ee-aba5-bbfb87cb88fa?embed=true&_g=(filters:!(),refreshInterval:(pause:!t,value:0),time:(from:now%2Fw,to:now%2Fw))&_a=(description:'To%20visualize%20metrics%20and%20logs',filters:!(),fullScreenMode:!f,options:(hidePanelTitles:!f,useMargins:!t),query:(language:kuery,query:''),timeRestore:!f,title:V1_metrics,viewMode:view)&show-time-filter=true"
       title="Kibana Dashboard"
       height="800" 
       width="100%"></iframe>

    </div>
  );
};

export default Dashboard;
